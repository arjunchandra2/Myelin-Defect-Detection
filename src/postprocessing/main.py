"""
postprocess.py 
- Remove bounding boxes which appear in consecutive planes as they are likely transverse axons
- Logic to remove defects can either be defects which are vesicle annotations in all z-planes or at least one z-plane
- Logic for IOU can either be all with respect to the base bbox or with respect to bbox in previous plane
- Saves annotations grouped together by logical defect
"""

import os
import copy
import argparse
import numpy as np

from scipy.io import loadmat
from scipy.io import savemat

from src import utils
from src.config import COLOR_ENCODING
from src.preprocessing.bbox import Bbox

# IOU threshold to remove bboxes
IOU_THRESHOLD = 0.5
# Max number of planes a vesicle can be in
MAX_PLANES = 2


def load_annotations_full(file_path):
    """
    - Same as utils function but includes confidence values added by model
    """
    annotations = utils.load_annotations(file_path)
    data = loadmat(file_path)
    annotations["conf"] = data["annotations"][0][6]

    return annotations


def add_bboxes(annotations):
    """
    - Add bboxes to Bbox class for tracking
    """
    for i in range(len(annotations["class_type"])):
        class_name = annotations["class_type"][i]
        z_plane = annotations["z_plane"][i]
        coords = annotations["bbox_coord"][i]
        conf = annotations["conf"][i]

        bbox = Bbox(
            coords[0], coords[1], coords[2], coords[3], z_plane, class_name, conf
        )


def save_cleaned_annotations(clean_annotations, im_path, save_path):
    """
    - Save the cleaned .mat file
    """
    class_names = []
    colors = []
    z_planes = []
    bboxes = []
    confs = []

    for bbox in clean_annotations:
        class_names.append(bbox.class_name)
        colors.append(list(map(float, COLOR_ENCODING[bbox.class_name])))
        z_planes.append([int(bbox.z_plane)])
        bboxes.append(list(map(float, bbox.get_coords())))
        confs.append([float(bbox.conf)])

    annotations = np.array(
        [im_path, "YOLOv8", class_names, colors, z_planes, bboxes, confs], dtype=object
    )

    mat_annotations = {"annotations": annotations}
    savemat(save_path, mat_annotations)


def all_vesicles(boxes):
    """
    - Returns True if all bboxes are vesicle detections, False otherwise
    """
    for bbox in boxes:
        if bbox.class_name != "Vesicle":
            return False

    return True


def main():
    """Main"""
    global IOU_THRESHOLD
    global MAX_PLANES

    parser = argparse.ArgumentParser(
        description="Sort bounding boxes. Can also remove vesicle detections in at least 3 consecutive planes"
    )

    # Required arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing .mat annotation files to be postprocessed",
    )

    # Optional arguments
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="IOU threshold to consider bboxes as overlapping",
    )

    parser.add_argument(
        "--max_planes",
        type=int,
        default=MAX_PLANES,
        help="Max number of planes a vesicle can be in",
    )

    parser.add_argument(
        "--remove_transverse_axons",
        action="store_true",
        help="Removes vesicle detections in > max_planes consecutive planes from the annotations",
    )

    args = parser.parse_args()
    # Set the IOU threshold
    IOU_THRESHOLD = args.iou_threshold
    # Set the max number of planes
    MAX_PLANES = args.max_planes

    postprocessed_dir = os.getcwd() + "/postprocessed"
    os.mkdir(postprocessed_dir)

    for file in os.listdir(args.data_path):
        if file.endswith(".mat"):

            mat_file = file
            im_path = mat_file[:-4]
            save_path = os.path.join(postprocessed_dir, mat_file)

            mat_path = os.path.join(args.data_path, mat_file)

            # reading .mat and adding bboxes to Bbox class
            annotations = load_annotations_full(mat_path)
            add_bboxes(annotations)

            num_planes = max(annotations["z_plane"])
            num_removed = 0

            # STEP 1: Remove vesicle detections in at least 3 consecutive z-planes

            # We will remove bounding boxes from here as we go
            cleaned_bboxes = copy.deepcopy(Bbox.bboxes_unseen)

            # iterate over all bounding boxes in order by z_plane
            for z in range(1, num_planes + 1):
                if z in Bbox.bboxes_unseen:
                    for current_bbox in Bbox.bboxes_unseen[z]:

                        if current_bbox in cleaned_bboxes[z]:
                            # we will measure IOU against the base bbox
                            base_bbox = current_bbox
                            overlap_bboxes = []
                            consecutive_planes = 1

                            for z_next in range(z + 1, num_planes):
                                # If there are any bboxes in the next plane
                                if z_next in Bbox.bboxes_unseen:
                                    # Look for any bboxes that overlap > threshold with base bbox
                                    # If there are multiple we will remove them all
                                    seen = False
                                    for candidate_bbox in Bbox.bboxes_unseen[z_next]:
                                        iou = utils.compute_iou(
                                            base_bbox.get_coords(),
                                            candidate_bbox.get_coords(),
                                        )
                                        if iou > IOU_THRESHOLD:
                                            overlap_bboxes.append(candidate_bbox)
                                            if not seen:
                                                consecutive_planes += 1
                                                seen = True

                                # Stop looping if there are no bboxes in the next z_plane
                                else:
                                    break

                                # If we did not find any bboxes which overlap > threshold with base bbox we stop
                                if not seen:
                                    break

                                # This sets the reference bbox for computing iou to the one in the most recent z-plane, useful for transverse axons b/c they
                                # move a little bit betwee z-planes
                                base_bbox = overlap_bboxes[-1]

                            # Remove bboxes if the argument is specified and bboxes are in more than MAX_PLANES and are all vesicle detections (if we haven't already removed them)
                            if (
                                args.remove_transverse_axons
                                and consecutive_planes > MAX_PLANES
                                and all_vesicles([current_bbox] + overlap_bboxes)
                            ):
                                if current_bbox in cleaned_bboxes[z]:
                                    cleaned_bboxes[z].remove(current_bbox)
                                    num_removed += 1
                                for bbox in overlap_bboxes:
                                    if bbox in cleaned_bboxes[bbox.z_plane]:
                                        cleaned_bboxes[bbox.z_plane].remove(bbox)
                                        num_removed += 1

            # STEP 2: Sort cleaned bboxes to save .mat file with logical defects grouped together

            # We will add the logical defects here as we go to save them together
            cleaned_bboxes_sorted = []

            # iterate over all bounding boxes in order by z_plane
            for z in range(1, num_planes + 1):
                if z in cleaned_bboxes:
                    for current_bbox in cleaned_bboxes[z]:
                        # we will measure IOU against the base bbox
                        base_bbox = current_bbox
                        overlap_bboxes = []
                        consecutive_planes = 1

                        for z_next in range(z + 1, num_planes):
                            # If there are any bboxes in the next plane
                            if z_next in cleaned_bboxes:
                                # Look for any bboxes that overlap > threshold with base bbox
                                # If there are multiple we will remove them all
                                seen = False
                                for candidate_bbox in cleaned_bboxes[z_next]:
                                    iou = utils.compute_iou(
                                        base_bbox.get_coords(),
                                        candidate_bbox.get_coords(),
                                    )
                                    if iou > IOU_THRESHOLD:
                                        overlap_bboxes.append(candidate_bbox)
                                        if not seen:
                                            consecutive_planes += 1
                                            seen = True

                            # Stop looping if there are no bboxes in the next z_plane
                            else:
                                break

                            # If we did not find any bboxes which overlap > threshold with base bbox we stop
                            if not seen:
                                break

                            # This sets the reference bbox for computing iou to the one in the most recent z-plane
                            base_bbox = overlap_bboxes[-1]

                        # Now we just save the logical defect together
                        if current_bbox not in cleaned_bboxes_sorted:
                            cleaned_bboxes_sorted.append(current_bbox)
                        for bbox in overlap_bboxes:
                            if bbox not in cleaned_bboxes_sorted:
                                cleaned_bboxes_sorted.append(bbox)

            print(f"Processing {mat_file}:")
            print(
                f"Number of annotations before: {len([item for sublist in Bbox.bboxes_unseen.values() for item in sublist])}"
            )
            print(
                f"Number of annotations after: {len([item for sublist in cleaned_bboxes.values() for item in sublist])}"
            )

            save_cleaned_annotations(cleaned_bboxes_sorted, im_path, save_path)
            print(f"Removed a total of {num_removed} bboxes")

            # Clear the Bbox class for the next file
            Bbox.bboxes_unseen = {}


if __name__ == "__main__":
    """Run from Command Line"""
    main()
