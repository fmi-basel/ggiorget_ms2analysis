"""
Spot linking for MS2 Image Analysis Pipeline
====================
This script takes a list of MS2 labeled images and mask images and linkes the spots over time
The general workflow includes:
1. data loading
2. linking spots, which includes
    clean up of double spots per cell
    gap closing between detected spots
    construction of non-bursting cell tracks

author: Jana Tuennermann
"""

# Let's start with loading all the packages and modules you need for this script
import argparse
import os

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from gapclosing import gap_closing


def main(spots_path, segmentation_path, path_output):
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(spots_path)[1]

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # 1. ------- Data loading--------
    df_spots = pd.read_csv(spots_path)

    mask_labelimagename = images_filename.replace('_spots.csv', '_label-image.tiff')
    mask_image = io.imread(os.path.join(segmentation_path, mask_labelimagename))
    mask_csvname = images_filename.replace('_spots.csv', '_label-image_tracks.csv')
    mask_track = pd.read_csv(os.path.join(segmentation_path, mask_csvname))

    # 2. ------- Linking --------
    # Generally speaking, I will track all cells, no matter if they showed spots or not.
    # For cells, with no spots, I read out the intensity in the center
    # For cells with spots/bursts, I need to close gabs. For this, I take the first/last know position of a spot and
    # interpolate the genes position based on lateral cell movement and cell deformation. Cell rotation is not
    # explicitly included, but is guessed by interpolating between the two know positions.
    print('Start linking')
    # Let's start by getting rid of more spurious spots: If more than one spot per frame per cell,
    # only keep the brightest spot
    df_spots = df_spots.groupby(['track_id', 'frame']).apply(lambda df: df.loc[df.mass.idxmax()],
                                                             include_groups=True).reset_index(drop=True)

    # Now, let's interpolate the position of the gene:
    # Here, semi-empty df with spot data and NaN for frames without spots, in here I write all the info I generate
    df_tracks = pd.merge(mask_track[['frame', 'track_id', 'parental_id']],
                         df_spots[['frame', 'track_id', 'x', 'y']], how='left')
    df_tracks['spotdetected'] = df_tracks['x'].notnull()

    # Find Cell IDs which (don't) show spots, careful: here the IDs refer to the unique ones
    cellids_nospots = np.setdiff1d(df_tracks['track_id'], df_spots['track_id'])
    cellids_spots = np.setdiff1d(df_tracks['track_id'], cellids_nospots)

    # If a cell shows no spots, take the center of mass of the cell as position
    df_nospots = mask_track[mask_track['track_id'].isin(cellids_nospots)]
    df_nospots = df_nospots.rename(columns={'centroid-y': 'y', 'centroid-x': 'x'})
    df_tracks = df_tracks.fillna(df_nospots)

    # Gab filling, including lateral cell movement and cell deformation
    for cell in tqdm(cellids_spots):
        # Do everything cell by cell, so get the cell label image, cell track, spot info etc
        df_cellspots = df_tracks[df_tracks['track_id'] == cell]
        cellmask_track = mask_track[mask_track['track_id'] == cell]
        cellmask_image = np.zeros(mask_image.shape)
        cellmask_image[:, :, :][mask_image == cell] = 1
        # actually fill the gabs
        df_cellspots = gap_closing(df_cellspots, 'x', 'y', cellmask_track, cellmask_image)
        df_tracks = df_tracks.fillna(df_cellspots)

    # 3. ------- Saving data --------
    df_tracks.to_csv(os.path.join(path_output, images_filename.replace('_spots.csv', '_tracks.csv')), index=False)
    print('Done :)')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--spots",
        type=str,
        default=None,
        required=True,
        help="Dataframe containing previously detected spots, requires absolute path",
    )
    parser.add_argument(
        "-is",
        "--input_segmentation",
        type=str,
        default=None,
        required=True,
        help="Directory in which segmentation images are saved, requires absolute path",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        default=None,
        required=True,
        help="Path to output directory, requires absolute path",
    )
    args = parser.parse_args()

    main(spots_path=args.spots, segmentation_path=args.input_segmentation, path_output=args.path_output)
