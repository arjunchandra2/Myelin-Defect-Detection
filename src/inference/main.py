"""
Running model inference and stitching predictions for .tif image into .mat format 
"""

import math
import time
import os
import numpy as np
import argparse
from tqdm import tqdm

import cv2
from scipy.io import savemat
from ultralytics import YOLO
from tifffile import TiffFile

from src.inference.annotation import Annotation
from src.config import COLOR_ENCODING, CLASS_NUM
from src import utils


# class encodings (inverse of CLASS_NUM)
CLASS_ENCODING = {v: k for k, v in CLASS_NUM.items()}

# batch size for inference
BATCH_SIZE = 32
# confidence threshold for detections (0-1)
CONFIDENCE_THRESHOLD = 0.30
# non max supression threshold (0-1)
NMS_THRESHOLD = 0.2
# window size: this is sliding window size
WINDOW_SIZE = 100
# window overlap for sliding window Should be (0-1)
WINDOW_OVERLAP = 0.5


def get_mat(im_path):
    """
    - Creates well-formatted dictionary for .mat annotations
    - Performs plane by plane NMS: https://gist.github.com/leandrobmarinho/26bd5eb9267654dbb9e37f34788486b5
    """
    class_names = []
    colors = []
    z_planes = []
    bboxes = []
    confs = []

    for z_plane in Annotation.annotations:
        bboxes_z = []
        confs_z = []
        cls_names_z = []

        for annotation in Annotation.annotations[z_plane]:
            for i in range(len(annotation.bboxes)):
                bbox = annotation.bboxes[i]
                bbox = [
                    float(bbox[0] + annotation.window_left),
                    float(bbox[1] + annotation.window_upper),
                    float(bbox[2]),
                    float(bbox[3]),
                ]
                bboxes_z.append(bbox)
                confs_z.append(float(annotation.confs[i]))
                cls_names_z.append(annotation.cls_names[i])

        indices = cv2.dnn.NMSBoxes(
            bboxes_z, confs_z, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
        )

        for index in indices:
            idx = index[0] if isinstance(index, (list, np.ndarray)) else index
            cls_name = cls_names_z[idx]
            class_names.append(cls_name)
            colors.append(list(map(float, COLOR_ENCODING[cls_name])))
            z_planes.append([int(z_plane)])
            bboxes.append(list(map(float, bboxes_z[idx])))
            confs.append([float(confs_z[idx])])

    # Final annotations stored as a NumPy object array
    annotations = np.array(
        [im_path, "YOLOv8", class_names, colors, z_planes, bboxes, confs], dtype=object
    )

    mat_annotations = {"annotations": annotations}

    return mat_annotations


def inference(model, image, grayscale, image_type):
    """
    - Image inference from local Yolov8 model
    - Runs on a full z-stack .tif image and saves predictions to .mat file
    """

    im_path = image

    with TiffFile(im_path) as tif:
        num_pages = len(tif.pages)

    if image_type == "rgb":
        load_image = utils.process_image_rgb_mc
    elif image_type == "grayscale":
        load_image = utils.process_image_grayscale_mc

    # Estimating total number of batches for full image
    num_z_planes = num_pages
    z_plane = load_image(im_path, 0)
    width, height = z_plane.size
    num_windows_x = (
        math.ceil((width - WINDOW_SIZE) / (WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP))
        + 1
    )
    num_windows_y = (
        math.ceil((height - WINDOW_SIZE) / (WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP))
        + 1
    )
    total_windows = num_windows_x * num_windows_y * num_z_planes
    total_batches = math.ceil(total_windows / BATCH_SIZE)

    pbar = tqdm(
        total=total_batches, desc=f"Running Inference on {im_path}", unit="batch"
    )

    for z in range(num_pages):

        z_plane = load_image(im_path, z)

        width, height = z_plane.size

        windows = []
        window_coords = []

        window_upper = 0
        window_lower = WINDOW_SIZE

        while window_upper < height:
            window_left = 0
            window_right = WINDOW_SIZE

            while window_left < width:
                window = z_plane.crop(
                    (window_left, window_upper, window_right, window_lower)
                )

                if grayscale:
                    window = utils.convert_to_grayscale(window)

                windows.append(window)
                window_coords.append((window_left, window_upper))

                # Run inference when batch is full
                if len(windows) == BATCH_SIZE:
                    pbar.update(1)
                    bboxes_all, cls_names_all, confs_all = get_yolo_predictions(
                        model, windows
                    )

                    for i in range(len(bboxes_all)):
                        bboxes = bboxes_all[i]
                        cls_names = cls_names_all[i]
                        confs = confs_all[i]
                        window_left, window_upper = window_coords[i]

                        if len(bboxes) > 0:
                            annotation = Annotation(
                                bboxes,
                                cls_names,
                                confs,
                                z + 1,
                                window_left,
                                window_upper,
                            )

                    windows = []
                    window_coords = []

                if window_right == width:
                    break

                window_left = window_left + WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP
                window_right = min(
                    width, window_right + WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP
                )

            if window_lower == height:
                break

            window_upper = window_upper + WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP
            window_lower = min(
                height, window_lower + WINDOW_SIZE - WINDOW_SIZE * WINDOW_OVERLAP
            )

        # Last batch is not batch size, just whatever is leftover
        if windows:
            bboxes_all, cls_names_all, confs_all = get_yolo_predictions(model, windows)

            for i in range(len(bboxes_all)):
                bboxes = bboxes_all[i]
                cls_names = cls_names_all[i]
                confs = confs_all[i]
                window_left, window_upper = window_coords[i]

                if len(bboxes) > 0:
                    annotation = Annotation(
                        bboxes,
                        cls_names,
                        confs,
                        z + 1,
                        window_left,
                        window_upper,
                    )

    # Create dictionary from Annotations
    mat_dict = get_mat(im_path)

    # Create .mat file and save
    savemat(im_path + ".mat", mat_dict)

    # Clear annotations for image
    Annotation.annotations = {}

    pbar.close()


def get_yolo_predictions(model, images):
    """
    - Batch inference from local Yolov8 model, images can be a list of images or a single image
    - YOLO accepted formats: https://docs.ultralytics.com/modes/predict/#inference-sources
    """
    # imgsz = (width, height), recommended to resize to (640,640) -> seems to work fine even for rectangular images
    # resizing maintains aspect ratio using rescale and pad and maintains multiple of 32 (network stride)
    results = model.predict(
        source=images,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        iou=NMS_THRESHOLD,
        device=os.getenv("CUDA_VISIBLE_DEVICES"),
        verbose=True,
    )

    bboxes_all = []
    cls_names_all = []
    confs_all = []

    # only a single result if no batch inference
    for result in results:
        preds = result.boxes.cpu().numpy()
        bboxes = preds.xyxy.astype("int32")
        cls_names = preds.cls
        confs = preds.conf

        # cls_id -> cls_name
        cls_names = [CLASS_ENCODING[int(cls_id)] for cls_id in cls_names]
        # xyxy -> xywh
        bboxes = [
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxes
        ]

        bboxes_all.append(bboxes)
        cls_names_all.append(cls_names)
        confs_all.append(confs)

    return bboxes_all, cls_names_all, confs_all


def main():
    global WINDOW_SIZE
    global WINDOW_OVERLAP
    global CONFIDENCE_THRESHOLD
    global NMS_THRESHOLD
    global BATCH_SIZE

    parser = argparse.ArgumentParser(
        description="Run inference on images using YOLOv8 model"
    )

    # Required arguments
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the directory contain .tif or .btf images",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the YOLOv8 model weights file",
    )

    parser.add_argument(
        "--image_type",
        required=True,
        choices=["rgb", "grayscale"],
        help="Type of the images (8-bit rgb or 16-bit grayscale)",
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=WINDOW_SIZE,
        help="Sliding window size for inference",
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=WINDOW_OVERLAP,
        help="Sliding window overlap for inference",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=NMS_THRESHOLD,
        help="Non-max suppression threshold",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="For RGB images, will apply (R,G,B) -> (G,G,G) (default: False)",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Update global variables with command-line arguments
    BATCH_SIZE = args.batch_size
    WINDOW_SIZE = args.window_size
    WINDOW_OVERLAP = args.window_overlap
    CONFIDENCE_THRESHOLD = args.conf_threshold
    NMS_THRESHOLD = args.nms_threshold

    model = YOLO(args.model_path)

    # timing
    start_time = time.perf_counter()
    for file in os.listdir(args.data_path):
        if file.endswith(".tif") or file.endswith(".btf"):
            image_path = os.path.join(args.data_path, file)

            if not os.path.exists(image_path + ".mat"):
                inference(model, image_path, args.grayscale, args.image_type)

    finish_time = time.perf_counter()
    elapsed = finish_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nInference finished in {minutes} minutes and {seconds} seconds.")


if __name__ == "__main__":
    """Run from Command Line"""
    main()
