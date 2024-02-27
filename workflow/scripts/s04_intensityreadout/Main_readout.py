"""
Intensity read-out for MS2 Image Analysis Pipeline (mask)
====================
This script takes 2D-time course (xyt) of MS2 labeled images and corresponding tracks and reads out the spot intensity
over time using a circular mask. It also reads out the intensity of the whole cell and GFP intensity.
The general workflow includes:
1. data loading, pre-processing (flat-field correction)
2. read-out of spot intensities
3. read-out whole cell intensity
4. calculate bleach corrected intensities
5. read-out GFP intensity

author: Jana Tuennermann
"""

# Let's start with loading all the packages and modules you need for this script
import argparse
import os
import sys

import numpy as np
import pandas as pd
from skimage import io

from circular_readout import circular_readout
from global_cell_readout import wholecell_readout_timeseries, wholecell_readout_singleframe


def flatfieldcorrection(image, flatfieldimage, darkimage):
    """
    flat-field correction using following approach: corrected image = image * (mean flat-field image) /flat-field image
    Args:
         image: (ndarray) image to be corrected
         flatfieldimage: (ndarray) flat-field image for correction
         darkimage: (ndarray) dark-image for correction (detector noise)

    Returns:
         image: (ndarray) corrected image
    """
    image = (image - darkimage) * np.mean(flatfieldimage - darkimage) / (flatfieldimage - darkimage)
    return image


def bleach_correction(trace_intensity, cell_intensity):
    """
    simple bleach correction using normalized cell intensity as a bleach factor
    Args:
         trace_intensity: (pd.Series) intensity trace to be corrected
         cell_intensity: (pd.Series) intensity trace of whole cell used for bleach correction

    Returns:
         corr_trace_intensity: (pd.Series) corrected intensity trace
    """
    norm_bleach_factor = cell_intensity / np.max(cell_intensity)
    corr_trace_intensity = trace_intensity / norm_bleach_factor
    return corr_trace_intensity


def main(image_path, tracks_path, mask_image_path, gfp_image_path, flatfield_path, path_output, spotdiameter):
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(image_path)[1]

    # 1. ------- Data loading and Pre-processing--------
    images_maxproj = io.imread(image_path)
    df_tracks = pd.read_csv(tracks_path)
    mask_image = io.imread(mask_image_path)
    gfp_image = io.imread(gfp_image_path)

    # Load images for flat-field correction. I use an automated approach to recognize the needed images, so I also
    # check if it works before continuing
    flatfield_images_list = os.listdir(flatfield_path)
    flatfield_cy5_filename = [s for s in flatfield_images_list if "568" in s and "mCherry-GFPCy5" in s]
    darkimage_cy5_filename = [s for s in flatfield_images_list if "Darkimage" in s and "mCherry-GFPCy5" in s]
    flatfield_gfp_filename = [s for s in flatfield_images_list if "488" in s and "GFP-Cy5mCherry" in s]
    darkimage_gfp_filename = [s for s in flatfield_images_list if "Darkimage" in s and "GFP-Cy5mCherry" in s]
    for image in [flatfield_cy5_filename, darkimage_cy5_filename, flatfield_gfp_filename, darkimage_gfp_filename]:
        if len(image) != 1:
            print('Automated selection for flat-flied images did not work. Either naming was incorrect or an '
                  'image is missing')
            sys.exit()
    flatfield_cy5_filename = flatfield_cy5_filename[0]
    darkimage_cy5_filename = darkimage_cy5_filename[0]
    flatfield_gfp_filename = flatfield_gfp_filename[0]
    darkimage_gfp_filename = darkimage_gfp_filename[0]

    flatfield_cy5_image = io.imread(os.path.join(flatfield_path, flatfield_cy5_filename))
    dark_cy5_image = io.imread(os.path.join(flatfield_path, darkimage_cy5_filename))
    flatfield_gfp_image = io.imread(os.path.join(flatfield_path, flatfield_gfp_filename))
    dark_gfp_image = io.imread(os.path.join(flatfield_path, darkimage_gfp_filename))

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # Flat-field correction
    images_corr = np.stack(
        [flatfieldcorrection(images_maxproj[i, ...], flatfield_cy5_image, dark_cy5_image) for i in
         range(images_maxproj.shape[0])],
        axis=0)

    images_gfp_corr = flatfieldcorrection(gfp_image, flatfield_gfp_image, dark_gfp_image)

    # 2. ------- MS2 intensity read-out (from flat-field corrected image) --------
    # The intensity is read/out using a circular mask around the position. The same estimated spot size is used as for
    # spot detection. Additionally, I read out the background intensity, using a ring around the circular mask. It's
    # seperated by a no of pixels (gap size), which I determined before.

    # Here the geometrics I will use
    ring_size = 1
    gap_size = 4
    total_size = spotdiameter + (ring_size + gap_size) * 2

    # Using a circular mask around the spot, read-out intensity
    spot_coordinates = df_tracks[['frame', 'y', 'x']].values.tolist()
    df_spot_intensity = circular_readout(images_corr, spot_coordinates, spotdiameter, ring_size, gap_size)
    df_tracks = df_tracks.merge(df_spot_intensity)

    # 3. ------- Whole cell read-out --------
    df_intensity_completecell = wholecell_readout_timeseries(images_corr, mask_image, spot_coordinates,
                                                             radius=total_size, excludearea=False)
    df_tracks = df_tracks.merge(df_intensity_completecell, how='left')

    # 4. ------- Bleach correction --------
    # based on the intensity of the whole cell, correct spot and local background intensity for bleaching. Then
    # calculate the corrected trace intensity (spot-local background)
    df_tracks['mean_spot_bleach'] = df_tracks.groupby('track_id').apply(
        lambda cell: bleach_correction(cell.mean_spot, cell.mean_completecell), include_groups=False).reset_index(
        level=0, drop=True)
    df_tracks['mean_localbackground_bleach'] = df_tracks.groupby('track_id').apply(
        lambda cell: bleach_correction(cell.mean_localbackground, cell.mean_completecell),
        include_groups=False).reset_index(level=0, drop=True)
    df_tracks['corr_trace'] = df_tracks['mean_spot_bleach'] - df_tracks['mean_localbackground_bleach']

    # 5. ------- GFP read-out --------
    # Using the first mask frame, read out the GFP intensity of the whole cell
    print('Reading out GFP')
    df_intensity_gfp = wholecell_readout_singleframe(images_gfp_corr, mask_image[0, :, :])
    df_intensity_gfp.columns = df_intensity_gfp.columns.str.replace('_completecell', '_gfp')
    df_intensity_gfp.drop(columns='area_gfp', inplace=True)
    df_tracks = df_tracks.merge(df_intensity_gfp, how='left')

    # 5. ------- Saving data --------
    df_tracks.to_csv(os.path.join(path_output, images_filename.replace('_MAX.tiff', '_tracks_intensity.csv')),
                     index=False)
    print('Done :)')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Input movie to be used, requires absolute path",
    )
    parser.add_argument(
        "-it",
        "--input_tracks",
        type=str,
        default=None,
        required=True,
        help="path to tracks dataframe corresponding to movie, requires absolute path",
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
        "-ig",
        "--input_gfpimage",
        type=str,
        default=None,
        required=True,
        help="path to gfp image corresponding to movie, requires absolute path",
    )
    parser.add_argument(
        "-if",
        "--input_flatfield",
        type=str,
        default=None,
        required=True,
        help="Directory in which images for flat-field correction are saved, requires absolute path",
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
        help="Estimated spot diameter for detection and int readout",
    )
    args = parser.parse_args()

    main(image_path=args.input_path, tracks_path=args.input_tracks, mask_image_path=args.input_segmentation_image,
         gfp_image_path=args.input_gfpimage, flatfield_path=args.input_flatfield, path_output=args.path_output,
         spotdiameter=args.spot_diameter)
