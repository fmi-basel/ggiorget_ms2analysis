"""
Post-processing for MS2 Image Analysis Pipeline (mask)
====================
This script takes intensity tracks of MS2 labeled images and performs following post-processing: filtering spot
detection column for single positives.
The general workflow includes:
1. data loading
2. single positive filtering

author: Jana Tuennermann
"""
import argparse
import os

import numpy as np
import pandas as pd
from skimage.morphology import remove_small_holes


def remove_positives(data, area_size=2):
    """
    From a bool signal trace, removes True values of certain length
    Args:
         data: (pd.Series) data trace to be processed
         area_size: (int) minimal length of consecutive True values to keep

    Returns:
         data_array_filtered: (pd.Series) filtered data trace
    """
    data_array = data.apply(lambda x: int(not x))
    data_array = np.array(data_array.tolist())
    data_array_filtered = remove_small_holes(data_array, area_threshold=area_size)
    data_array_filtered = data_array_filtered.astype(bool)
    return data_array_filtered


def main(tracks_path, min_burstlength, path_output):
    # 1. ------- Data loading --------
    df_tracks = pd.read_csv(tracks_path)
    filename = os.path.basename(tracks_path)
    images_filename = filename.replace('_tracks_intensity.csv', '_MAX.tiff')
    # 2. ------- remove single pos --------
    # remove single pos values from spot detection column
    df_tracks['spotdetected_filtered'] = float('nan')
    if not df_tracks.empty:
        df_tracks['spotdetected_filtered'] = df_tracks.groupby('track_id').apply(
            lambda cell: remove_positives(cell.spotdetected, area_size=min_burstlength),
            include_groups=False).reset_index(
            level=0, drop=True)

    # 3. ------- Saving data --------
    # also add some useful naming columns
    df_tracks['filename'] = images_filename
    df_tracks['clone'] = float('nan')
    if not df_tracks.empty:
        df_tracks['clone'] = df_tracks['filename'].str.rsplit(pat='_', n=-1, expand=True)[3]
    df_tracks.to_csv(os.path.join(path_output, filename.replace('_intensity.csv', '_postprocessed.csv')), index=False)


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-it",
        "--input_tracks",
        type=str,
        default=None,
        required=True,
        help="path to tracks dataframe corresponding to movie, requires absolute path",
    )
    parser.add_argument(
        "-mb",
        "--min_burstlength",
        type=int,
        default=None,
        required=True,
        help="minimun length of a burst",
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

    main(tracks_path=args.input_tracks, min_burstlength=args.min_burstlength, path_output=args.path_output)
