"""
Spot detection for MS2 Image Analysis Pipeline
====================
This script takes 2D-time course (xyt) of MS2 labeled images and detects spots
The general workflow includes:
1. data loading, pre-processing (background subtraction)
2. spot detection and assignment to cells using the label mask
3. spot filtering based on spot size, intensity and cell co-localisation

author: Jana Tuennermann
"""

# Let's start with loading all the packages and modules you need for this script
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
from skimage import io


def local_backgroundsubtraction(image, pixelsize):
    """
    local background subtraction using a round kernel and a defined pixel size
    Args:
         image: (ndarray) image to be background subtracted
         pixelsize: (int) kernel size in pixels, need to be an uneven number

    Returns:
         image: (ndarray) background subtracted image
    """
    from skimage.morphology import white_tophat
    from skimage.morphology import disk
    image = white_tophat(image, disk(pixelsize))
    return image


def main(image_path, mask_image_path, path_output, spotdiameter, threshold):
    tp.quiet()
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(image_path)[1]

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # Make a sub-folder for some quality assessment plots
    if not os.path.exists(os.path.join(path_output, 'plots')):
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(path_output, 'plots'))

    # 1. ------- Data loading and Pre-processing--------
    images_maxproj = io.imread(image_path)
    mask_image = io.imread(mask_image_path)

    # local background subtraction of images, works better for spot detection later
    images_sub = np.stack(
        [local_backgroundsubtraction(images_maxproj[i, ...], pixelsize=5) for i in range(images_maxproj.shape[0])],
        axis=0)

    print('Preprocessing Done, start spot detection')

    # 2. ------- spot detection --------
    # trackpy first uses a bandpass filter and locates the spot using the centroid-finding algorithm from Crocker-Grier
    # It is a threshold-based approach

    # Detection
    # JF646 wash: diameter 7, minmass 1400 (Max_sub)
    # JF646 no wash: diameter 7, minmass 2000 (Max_sub)
    # JF549 wash: diameter 9, minmass 3900 (Max_sub)
    # JF549 no wash: diameter 9, minmass 4300 (Max_sub)
    # JF549 wash 5 clones: diameter 9, minmass 4600 (Max_sub)
    df_spots = tp.batch(images_sub, diameter=spotdiameter, minmass=threshold)

    # the subpx_bias is a function to see if the diameter you chose makes sense, the resulting histogram should be flat
    # tp.subpx_bias(df_spots)
    # plt.show()

    # 3. ------- Spot filtering--------
    # Clean up for spurious spots not assigned to cells
    # Assign spots to cells (Label image ID)
    t = df_spots['frame'].astype(np.int64)
    y = df_spots['y'].astype(np.int64)
    x = df_spots['x'].astype(np.int64)
    df_spots['track_id'] = mask_image[t, y, x]
    # Let's get rid of spots which are not assigned to cells
    df_spots = df_spots[df_spots.track_id != 0]

    # I have some spots coming from camera issues, they can be small, but super intense, hence I use thresholding to
    # remove them. For quality control, I want to plot the spot properties I thresholded on and save them automatically
    # in the output folder
    threshold_size = 1.29  # default 1.19 for JF646, 1.29 for JF549
    threshold_mass = 30000  # default 21000 for JF646, 30000 for JF549
    # Create the plot and save it
    plt.scatter(df_spots['mass'], df_spots['size'])
    plt.hlines(y=threshold_size, xmin=min(df_spots['mass']), xmax=max(df_spots['mass']), colors='r', linestyles='--')
    plt.vlines(x=threshold_mass, ymin=min(df_spots['size']), ymax=max(df_spots['size']), colors='r', linestyles='--')
    plt.ylabel('Size')
    plt.xlabel('Mass')
    plt.title(images_filename, fontsize=5)
    plt.savefig(os.path.join(path_output, 'plots', images_filename.replace('_MAX.tiff', '_Spot-Filter.pdf')))
    # plt.show()
    plt.close('all')

    # Remove the camera error spots by thresholding
    df_spots = df_spots[df_spots['mass'] <= threshold_mass]
    df_spots = df_spots[df_spots['size'] >= threshold_size]

    # save spots
    df_spots.to_csv(os.path.join(path_output, images_filename.replace('_MAX.tiff', '_spots.csv')), index=False)
    print('Done :)')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input movie to be used, requires absolute path",
    )
    parser.add_argument(
        "-is",
        "--input_segmentation_image",
        type=str,
        default=None,
        required=True,
        help="Segmentation images to be used, requires absolute path",
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
        "-d",
        "--spot_diameter",
        type=int,
        default=9,
        required=True,
        help="Estimated spot diameter for detection",
    )
    parser.add_argument(
        "-t",
        "--spot_threshold",
        type=int,
        default=4600,
        required=True,
        help="Threshold for detection",
    )
    args = parser.parse_args()

    main(image_path=args.input, mask_image_path=args.input_segmentation_image, path_output=args.path_output,
         spotdiameter=args.spot_diameter, threshold=args.spot_threshold)
