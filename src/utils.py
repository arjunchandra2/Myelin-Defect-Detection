"""
utils.py
"""

from tifffile import TiffFile
from scipy.io import loadmat
from PIL import Image
from skimage import io
import numpy as np


def load_annotations(file_path):
    """
    - Read in and format annotations from .mat file into dictionary
    - Keep numpy datatypes
    """

    data = loadmat(file_path)
    class_type = data["annotations"][0][2]
    # strip whitespace
    for i in range(len(class_type)):
        class_type[i] = class_type[i].strip()

    z_plane = data["annotations"][0][4]
    z_plane = z_plane.flatten()

    bbox_coord = data["annotations"][0][5]

    assert len(class_type) == len(z_plane) == len(bbox_coord)

    annotations_dict = {
        "class_type": class_type,
        "z_plane": z_plane,
        "bbox_coord": bbox_coord,
    }

    return annotations_dict


def process_image_grayscale(image_path):
    """
    - Returns list of Pillow images for each frame for 16-bit grayscale images
    - Normalizes each frame separately
    """
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []

    # Normalize each frame separately
    for img in im_frames:
        frame_min = np.min(img)
        frame_max = np.max(img)

        # Normalize the frame to range [0, 255]
        img_normalized = (img - frame_min) / (frame_max - frame_min) * 255
        img_normalized = img_normalized.astype(np.uint8)  # Convert to uint8

        # Convert the normalized frame to a PIL image
        image = Image.fromarray(img_normalized, mode="L")
        pil_frames.append(image)

    return pil_frames


def process_image_grayscale_mc(image_path, z_index):
    """
    - Loads a single 16-bit grayscale slice from a BigTIFF image
    - Normalizes to 8-bit and returns as a Pillow image
    """
    with TiffFile(image_path) as tif:
        img = tif.pages[z_index].asarray()

    # Normalize the frame to range [0, 255]
    frame_min = np.min(img)
    frame_max = np.max(img)

    if frame_max > frame_min:
        img_normalized = (img - frame_min) / (frame_max - frame_min) * 255
    else:
        img_normalized = np.zeros_like(img, dtype=np.uint8)

    img_normalized = img_normalized.astype(np.uint8)

    return Image.fromarray(img_normalized, mode="L")


def process_image_rgb(image_path):
    """
    - Returns list of Pillow images for each frame for 8-bit RGB images
    """
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []
    # skimage -> PIL for easier cropping and displaying
    for im in im_frames:
        image = Image.fromarray(im, mode="RGB")
        pil_frames.append(image)

    return pil_frames


def process_image_rgb_mc(image_path, z_index):
    """
    - Loads a single 8-bit RGB slice from a BigTIFF image
    - Returns as a Pillow image
    """
    with TiffFile(image_path) as tif:
        img = tif.pages[z_index].asarray()

    return Image.fromarray(img, mode="RGB")


def swap_channels(image):
    """
    - Applies the following transformation to image channels and returns new image
    (R,G,B) -> (R-1.4B, G, B)
    - 4.5s for 10k images
    """
    channels = np.array(image)
    # apply transformation
    channels[:, :, 0] = abs(channels[:, :, 0] - 1.4 * channels[:, :, 2])
    # keep values in valid range
    channels = channels.clip(0, 255)

    return Image.fromarray(channels)


def swap_channels_lib(image):
    """
    - Same as swap_channels but using PIL library calls
    - 6.5s for 10k images
    """
    r, g, b = image.split()
    r = abs(np.asarray(r) - 1.4 * np.asarray(b))
    r = Image.fromarray(r.clip(0, 255).astype(np.uint8))

    return Image.merge("RGB", (r, g, b))


def convert_to_grayscale(image):
    """
    - Converts the RGB image to grayscale by copying the G channel to R and B
    (R,G,B) -> (G, G, G)
    """
    channels = np.array(image)
    # apply transformation
    channels[:, :, 0] = channels[:, :, 1]
    channels[:, :, 2] = channels[:, :, 1]

    return Image.fromarray(channels)


def compute_iou(bbox1, bbox2):
    """
    - Computes the Intersection over Union (IoU) of two bounding boxes
    - Bounding box format should be: (x_tl, y_tl, width, height)
    """

    # Convert the bounding box coordinates to a larger data type
    x1_tl, y1_tl, w1, h1 = map(np.int32, bbox1)
    x1_br, y1_br = x1_tl + w1, y1_tl + h1

    x2_tl, y2_tl, w2, h2 = map(np.int32, bbox2)
    x2_br, y2_br = x2_tl + w2, y2_tl + h2

    # Calculate the intersection coordinates
    inter_x_tl = max(x1_tl, x2_tl)
    inter_y_tl = max(y1_tl, y2_tl)
    inter_x_br = min(x1_br, x2_br)
    inter_y_br = min(y1_br, y2_br)

    # Compute the area of the intersection rectangle
    inter_width = max(0, inter_x_br - inter_x_tl)
    inter_height = max(0, inter_y_br - inter_y_tl)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Compute the union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Compute IoU (intersection over union)
    iou = inter_area / union_area if union_area != 0 else 0

    return iou
