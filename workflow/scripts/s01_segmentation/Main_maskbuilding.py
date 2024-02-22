"""
Segmentation for MS2 Image Analysis Piepeline
====================
This script takes 3D-images (tyx) of MS2 transcriptional spots and segments the nuclei using stardist and a linear
assignment problem (LAP) approach for tracking.
The general workflow includes:
1. data loading
2. nuclei segmentation using stardist
3. filtering cells by size
4. tracking of nuclei, removing short tracks
5. saving the results

author: Jana Tuennermann
"""

# import packages
import argparse
import os

import numpy as np
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border

from mask_tracking_LAP import mask_tracking_lap
from stardist_timeseries import stardist_timeseries

# Run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(path_images, path_output, min_tracklength, min_cellsize):
    # load image
    filename = os.path.basename(path_images)
    images = io.imread(path_images)

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # run stardist:
    # Default threshold arguments for the '2D_versatile_fluo' model are prob_thresh=0.48 (probability threshold),
    # nms_thresh=0.3 (overlap threshold); This can be adjusted based on needs and should be adjusted for different
    # models (I used a parameter sweep to find these values)
    # 10s movies: resizevalue = 128, prob_thresh = 0.68, nms_thresh = 0.3) Halo-JF646 wash
    # 70s movies: resizevalue = 128, prob_thresh = 0.28, nms_thresh = 0.3) Halo-JF646 wash
    # 30s movies: resizevalue = 128, prob_thresh = 0.58, nms_thresh = 0.3) Halo-JF646 wash
    # 30s movies: resizevalue = 128, prob_thresh = 0.38, nms_thresh = 0.3) Halo-JF646 no wash
    # 30s movies: resizevalue = 128, prob_thresh = 0.58, nms_thresh = 0.3) Halo-JF646 no exchange
    # 30s movies: resizevalue = 128, prob_thresh = 0.48, nms_thresh = 0.3) Halo-JF549
    labels = stardist_timeseries(images, stardist_prob_thresh=0.48, stardist_nms_thresh=0.3, scale_factor=0.33)

    # apply threshold on minimal cell size
    labels = np.asarray(
        [remove_small_objects(labels[frame, ...], min_size=min_cellsize) for frame in range(labels.shape[0])])

    # remove border touching cells
    labels = np.asarray([clear_border(labels[frame, ...]) for frame in range(labels.shape[0])])

    # tracking of the labels
    new_labels, new_track_df = mask_tracking_lap(labels, min_tracklength=min_tracklength)

    # save results: label image and tracking data
    io.imsave(os.path.join(path_output, filename.replace('_MAX.tiff', '_label-image.tiff')), new_labels,
              check_contrast=False, compression="zstd")
    new_track_df.to_csv(os.path.join(path_output, filename.replace('_MAX.tiff', '_label-image_tracks.csv')),
                        index=False)
    print('Done')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--path_input",
        type=str,
        default=None,
        required=True,
        help="Input movie to be used, requires absolute path",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        default=None,
        required=True,
        help="Path to output directory, requires absolute path",
    )
    parser.add_argument(
        "-t",
        "--min_tracklength",
        type=int,
        default=60,
        required=False,
        help="minimal number of consecutive frames to be considered for tracking",
    )
    parser.add_argument(
        "-c",
        "--min_cellsize",
        type=int,
        default=2000,
        required=False,
        help="minimal size of a cell in pixel",
    )
    args = parser.parse_args()

    main(path_images=args.path_input, path_output=args.path_output, min_tracklength=args.min_tracklength,
         min_cellsize=args.min_cellsize)
