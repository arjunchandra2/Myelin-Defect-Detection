"""
main.py - Cropping bounding boxes and creating new directory with images and corresponding annotations in YOLO format
"""

import random
import os
import time
import argparse
import numpy as np

from src.preprocessing.bbox import Bbox
from src.config import CLASS_NUM
from src import utils

# Keep track of number of images created in train and val, num bboxes missed
NUM_VAL = 0
NUM_TRAIN = 0
NUM_MISSED = 0

# Parameter to set for cropping
WINDOW_SIZE = 100

# Augmentation parameters
AUGMENTATION = False
NUM_CROPS = 2


def add_bboxes(annotations):
    """
    - Create bbox objects for all of the bounding boxes in each (3D) image
    - All of the bboxes for the image will be stored within the class
    - Can optionally specify a file with only clean defects
    """

    for i in range(len(annotations["class_type"])):
        class_name = annotations["class_type"][i]
        z_plane = annotations["z_plane"][i]
        coords = annotations["bbox_coord"][i]

        bbox = Bbox(coords[0], coords[1], coords[2], coords[3], z_plane, class_name)


def get_overlaps(left, upper, right, lower, z_plane):
    """
    - Returns list of bounding boxes exceeding overlap threshold for given window in given plane
    - Crops BBoxes as necessary
    """
    overlaps = []

    for bbox in Bbox.bboxes_unseen[z_plane]:
        # bbox is in the window - check overlap
        if left <= bbox.top_left_x <= right and upper <= bbox.top_left_y <= lower:
            total_area = bbox.width * bbox.height

            if bbox.top_left_x + bbox.width > right:
                width_inside = right - bbox.top_left_x
            else:
                width_inside = bbox.width
            if bbox.top_left_y + bbox.height > lower:
                height_inside = lower - bbox.top_left_y
            else:
                height_inside = bbox.height

            area_inside = width_inside * height_inside

            if area_inside / total_area > Bbox.OVERLAP_THRESHOLD:
                # crop bbox and add to overlap set
                bbox.width = width_inside
                bbox.height = height_inside
                overlaps.append(bbox)

    return overlaps


def save_annotations_yolo(left, upper, bboxes, data_save_path, z, i, theta=0):
    """
    - Save annotations in yolo format .txt file matching image path
    - optional theta parameter to rotate bounding boxes (used for augmentation)
    """

    # change extension
    file_path = data_save_path[:-3] + "(" + str(z + 1) + "_" + str(i) + ")" + ".txt"

    f = open(file_path, "w")

    for bbox in bboxes:
        # Skip background class bboxes
        if bbox.class_name == "Background":
            continue
        c_id = CLASS_NUM[bbox.class_name]

        cx = bbox.center_x()
        cy = bbox.center_y()
        width = bbox.width
        height = bbox.height

        # normalize coordinates
        cx_n = (cx - left) / WINDOW_SIZE
        cy_n = (cy - upper) / WINDOW_SIZE
        width_n = width / WINDOW_SIZE
        height_n = height / WINDOW_SIZE

        # rotate bounding box if rotation is specified
        if theta != 0:
            # translate (cx_n, cxy_n) to coordinate system with origin at center
            cx_coordinate = cx_n - 0.5
            cy_coordinate = 0.5 - cy_n
            # counterclockwise rotation by theta
            cx_rotated = cx_coordinate * np.cos(
                np.deg2rad(theta)
            ) - cy_coordinate * np.sin(np.deg2rad(theta))
            cy_rotated = cx_coordinate * np.sin(
                np.deg2rad(theta)
            ) + cy_coordinate * np.cos(np.deg2rad(theta))
            # undo coordinate translation
            cx_n = 0.5 + cx_rotated
            cy_n = 0.5 - cy_rotated

            # swap width and height of bbox if needed
            if theta == 90 or theta == 270:
                temp_width = width_n
                width_n = height_n
                height_n = temp_width

        entry = (
            str(c_id)
            + " "
            + str(cx_n)
            + " "
            + str(cy_n)
            + " "
            + str(width_n)
            + " "
            + str(height_n)
            + "\n"
        )
        f.write(entry)

    f.close()


def remove_blurry(frames):
    """
    - Removes bounding boxes from planes 1-7 and 22-25 since they are too blurry for annotations
    and only show up in the data due to bugs in original annotation software
    - z plane is one-indexed
    """

    for z in range(1, 8):
        if z in Bbox.bboxes_unseen:
            Bbox.bboxes_unseen[z] = []

    for z in range(len(frames) - 3, len(frames) + 1):
        if z in Bbox.bboxes_unseen:
            Bbox.bboxes_unseen[z] = []


def crop_bboxes(
    frames,
    im_save_path,
    data_save_path,
    remove_blurry_flag=False,
    grayscale=False,
    background_frac=1.0,
):
    """
    - Crop images around bounding box in each frame as we go and check for overlap
    - Z-plane is one-indexed as opposed to frames array
    - Images are saved to working directory with z_plane and crop number
    - Remove bounding boxes from first few and last few planes to clean up buggy annotations
    """
    global NUM_MISSED
    global NUM_VAL
    global NUM_TRAIN

    # remove blurry frames to ensure clean annotations
    if remove_blurry_flag:
        remove_blurry(frames)

    for z in range(len(frames)):
        if z + 1 in Bbox.bboxes_unseen:
            # crop number
            i = 0

            # loop over all bboxes in current plane without removal
            for current_bbox in Bbox.bboxes_unseen[z + 1]:
                # skip background class bboxes
                if current_bbox.class_name == "Background":
                    if random.random() > background_frac:
                        continue

                # window size is too small to capture bbox - we just skip (can occur due to buggy annotations)
                if (
                    current_bbox.width > WINDOW_SIZE
                    or current_bbox.height > WINDOW_SIZE
                ):
                    NUM_MISSED += 1
                    continue

                # set random cropping bounds
                if current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width > 0:
                    leftx = current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width
                else:
                    leftx = 0
                if current_bbox.top_left_x + WINDOW_SIZE > frames[z].size[0]:
                    rightx = current_bbox.top_left_x - (
                        current_bbox.top_left_x + WINDOW_SIZE - frames[z].size[0]
                    )
                else:
                    rightx = current_bbox.top_left_x

                if current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height > 0:
                    bottomy = (
                        current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height
                    )
                else:
                    bottomy = 0
                if current_bbox.top_left_y + WINDOW_SIZE > frames[z].size[1]:
                    topy = current_bbox.top_left_y - (
                        current_bbox.top_left_y + WINDOW_SIZE - frames[z].size[1]
                    )
                else:
                    topy = current_bbox.top_left_y

                left = random.randint(leftx, rightx)
                upper = random.randint(bottomy, topy)
                right = left + WINDOW_SIZE
                lower = upper + WINDOW_SIZE

                # get all overlapping bboxes in current window
                overlap_bboxes = get_overlaps(left, upper, right, lower, z + 1)

                cropped_im = frames[z].crop((left, upper, right, lower))
                if grayscale:
                    cropped_im = utils.convert_to_grayscale(cropped_im)

                save_path = (
                    im_save_path[:-3] + "(" + str(z + 1) + "_" + str(i) + ")" + ".png"
                )
                print("Saving image......." + save_path)
                cropped_im.save(save_path)

                # save annotations in yolo format
                save_annotations_yolo(left, upper, overlap_bboxes, data_save_path, z, i)

                i += 1

                if "valid" in save_path:
                    NUM_VAL += 1
                else:
                    NUM_TRAIN += 1

    # remove all bounding boxes from Bbox class after the image has been processed
    Bbox.bboxes_unseen = {}


def crop_bboxes_aug(
    frames,
    im_save_path,
    data_save_path,
    remove_blurry_flag=False,
    grayscale=False,
    background_frac=1.0,
):
    """
    - Same as crop_bboxes but with augmentation applied (see info.txt)
    - bboxes are rotated by save_annotations_yolo
    """
    global NUM_MISSED
    global NUM_TRAIN

    # remove blurry frames to ensure clean annotations
    if remove_blurry_flag:
        remove_blurry(frames)

    for z in range(len(frames)):
        if z + 1 in Bbox.bboxes_unseen:
            # crop number
            i = 0

            # loop over all bboxes in current plane without removal
            for current_bbox in Bbox.bboxes_unseen[z + 1]:
                # skip background class bboxes
                if current_bbox.class_name == "Background":
                    if random.random() > background_frac:
                        continue

                # window size is too small to capture bbox - we just skip (can occur due to buggy annotations)
                if (
                    current_bbox.width > WINDOW_SIZE
                    or current_bbox.height > WINDOW_SIZE
                ):
                    NUM_MISSED += 1
                    continue

                # set random cropping bounds
                if current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width > 0:
                    leftx = current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width
                else:
                    leftx = 0
                if current_bbox.top_left_x + WINDOW_SIZE > frames[z].size[0]:
                    rightx = current_bbox.top_left_x - (
                        current_bbox.top_left_x + WINDOW_SIZE - frames[z].size[0]
                    )
                else:
                    rightx = current_bbox.top_left_x

                if current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height > 0:
                    bottomy = (
                        current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height
                    )
                else:
                    bottomy = 0
                if current_bbox.top_left_y + WINDOW_SIZE > frames[z].size[1]:
                    topy = current_bbox.top_left_y - (
                        current_bbox.top_left_y + WINDOW_SIZE - frames[z].size[1]
                    )
                else:
                    topy = current_bbox.top_left_y

                # AUGMENTATION:
                # multiple random croppings around bounding box
                for n in range(NUM_CROPS):
                    # choose random window bounds
                    left = random.randint(leftx, rightx)
                    upper = random.randint(bottomy, topy)
                    right = left + WINDOW_SIZE
                    lower = upper + WINDOW_SIZE

                    overlap_bboxes = get_overlaps(left, upper, right, lower, z + 1)

                    cropped_im = frames[z].crop((left, upper, right, lower))
                    if grayscale:
                        cropped_im = utils.convert_to_grayscale(cropped_im)

                    # regular and transformed/swapped channels
                    for channels in ["original", "swapped"]:
                        if channels == "swapped":
                            cropped_im = utils.swap_channels(cropped_im)

                        # 4 orientations
                        for theta in [0, 90, 180, 270]:
                            cropped_im_rotated = cropped_im.rotate(theta)

                            save_path = (
                                im_save_path[:-3]
                                + "("
                                + str(z + 1)
                                + "_"
                                + str(i)
                                + ")"
                                + ".png"
                            )
                            print("Saving image......." + save_path)
                            cropped_im_rotated.save(save_path)

                            # MODIFY ANNOTATION FORMAT HERE
                            save_annotations_yolo(
                                left, upper, overlap_bboxes, data_save_path, z, i, theta
                            )

                            i += 1
                            NUM_TRAIN += 1

    # remove all bounding boxes from Bbox class after the image has been processed
    Bbox.bboxes_unseen = {}


def main():
    global WINDOW_SIZE
    global AUGMENTATION
    global NUM_CROPS

    parser = argparse.ArgumentParser(
        description="Crop bounding boxes and create new directory with images and corresponding annotations in YOLO format"
    )

    # Required arguments
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the directory contain .tif images and .mat annotation files",
    )
    parser.add_argument(
        "--val_images", required=True, nargs="+", help="Filenames of validation images"
    )

    parser.add_argument(
        "--image_type",
        required=True,
        choices=["rgb", "grayscale"],
        help="Type of the images (8-bit rgb or 16-bit grayscale)",
    )

    # Optional arguments
    parser.add_argument(
        "--seed", type=int, default=18, help="Random seed (default: 18)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size for cropping (default: 100)",
    )
    parser.add_argument(
        "--background_frac",
        type=float,
        default=1.0,
        help="Fraction of background images to use (default: 1.0)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable augmentation (default: False)",
    )
    parser.add_argument(
        "--num_crops",
        type=int,
        default=NUM_CROPS,
        help="Number of crops for augmentation (default: 2)",
    )
    parser.add_argument(
        "--remove_blurry",
        action="store_true",
        help="Removes first 7 and last 4 z-planes (default: False)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="For RGB images, will apply (R,G,B) -> (G,G,G) (default: False)",
    )

    args = parser.parse_args()

    WINDOW_SIZE = args.window_size
    AUGMENTATION = args.augmentation
    NUM_CROPS = args.num_crops
    random.seed(args.seed)

    # timing
    start_time = time.perf_counter()

    # This should be deleted manually before running the script
    results_dir = os.getcwd() + "/results"

    # create directory tree
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, "train"))
    os.mkdir(os.path.join(results_dir, "train", "images"))
    os.mkdir(os.path.join(results_dir, "train", "labels"))
    os.mkdir(os.path.join(results_dir, "valid"))
    os.mkdir(os.path.join(results_dir, "valid", "images"))
    os.mkdir(os.path.join(results_dir, "valid", "labels"))

    for file in os.listdir(args.data_path):
        if file.endswith(".tif"):

            image_path = os.path.join(args.data_path, file)
            data_path = image_path + ".mat"

            # if the file has been annotated then we crop
            if os.path.exists(data_path):
                # Read in image and store z_stack in array of PIL objects
                if args.image_type == "rgb":
                    im_frames = utils.process_image_rgb(image_path)
                elif args.image_type == "grayscale":
                    im_frames = utils.process_image_grayscale(image_path)

                # reading .mat and adding bboxes to Bbox class
                annotations = utils.load_annotations(data_path)
                add_bboxes(annotations)

                if file in args.val_images:
                    image_save_path = os.path.join(results_dir, "valid", "images", file)
                    data_save_path = os.path.join(results_dir, "valid", "labels", file)

                    # no augmentation for validation images
                    crop_bboxes(
                        im_frames,
                        image_save_path,
                        data_save_path,
                        args.remove_blurry,
                        args.grayscale,
                    )
                else:
                    image_save_path = os.path.join(results_dir, "train", "images", file)
                    data_save_path = os.path.join(results_dir, "train", "labels", file)

                    # crop and save bboxes using PIL img array 'frames' and Bbox class
                    if AUGMENTATION:
                        crop_bboxes_aug(
                            im_frames,
                            image_save_path,
                            data_save_path,
                            args.remove_blurry,
                            args.grayscale,
                            args.background_frac,
                        )
                    else:
                        crop_bboxes(
                            im_frames,
                            image_save_path,
                            data_save_path,
                            args.remove_blurry,
                            args.grayscale,
                            args.background_frac,
                        )

    finish_time = time.perf_counter()

    print()
    print(
        "Succesfully created dataset in ~", (finish_time - start_time) // 60, "minutes."
    )
    print("Dataset size:")
    print("Training set", NUM_TRAIN, "images")
    print("Validation set", NUM_VAL, "images")
    print(
        "A total of",
        NUM_MISSED,
        "bounding boxes could not be cropped with a window size of",
        WINDOW_SIZE,
    )
    print(
        "A total of",
        Bbox.BBOXES_REMOVED,
        "bounding boxes were removed as duplicates",
    )


if __name__ == "__main__":
    """Run from Command Line"""
    main()
