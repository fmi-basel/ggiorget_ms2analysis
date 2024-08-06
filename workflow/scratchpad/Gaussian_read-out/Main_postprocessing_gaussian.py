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
    data_array = data.astype(bool)
    data_array = data_array.apply(lambda x: not x)
    data_array_filtered = remove_small_holes(data_array, area_threshold=area_size)
    data_array_filtered = ~data_array_filtered.astype(bool)
    return data_array_filtered


def main(tracks_path, random_tracks_path, min_burstlength, path_output):
    # 1. ------- Data loading --------
    df_tracks = pd.read_csv(tracks_path)
    df_tracks_random = pd.read_csv(random_tracks_path)

    filename = os.path.basename(tracks_path)
    images_filename = filename.replace('_tracks_intensity.csv', '_MAX.tiff')

    read_out = 'intensity_corr'

    # 2. ------- burst calling ----------
    # calculate a dynamic treshold based on the random background intensities
    df_tracks_random = df_tracks_random.pivot(index=['frame', 'track_id'], columns='set', values=read_out)
    df_tracks_random['mean_random'] = df_tracks_random.mean(axis=1, numeric_only=True)
    df_tracks_random['std_random'] = df_tracks_random.std(axis=1, numeric_only=True)
    df_tracks_random['threshold'] = df_tracks_random['mean_random'].mean() + 5 * df_tracks_random['std_random'].mean()
    df_tracks_random.reset_index(inplace=True)
    # apply threshold
    df_tracks = df_tracks.merge(df_tracks_random[['track_id', 'frame', 'mean_random', 'std_random', 'threshold']],
                                how='left')
    df_tracks['burst_threshold'] = df_tracks[read_out] > df_tracks['threshold']

    # 2. ------- remove single pos --------
    # remove single pos values from spot detection column
    df_tracks['burst_threshold_filtered'] = float('nan')
    if not df_tracks.empty:
        df_tracks['burst_threshold_filtered'] = df_tracks.groupby('track_id').apply(
            lambda cell: remove_positives(cell.burst_threshold, area_size=min_burstlength),
            include_groups=False).reset_index(
            level=0, drop=True)

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
    df_tracks.to_csv(os.path.join(path_output, filename.replace('_intensity.csv', f'_{read_out}_postprocessed.csv')), index=False)


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
        "-itr",
        "--input_tracks_random",
        type=str,
        default=None,
        required=True,
        help="path to random tracks dataframe corresponding to movie, requires absolute path",
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

    main(tracks_path=args.input_tracks, random_tracks_path=args.input_tracks_random,
         min_burstlength=args.min_burstlength, path_output=args.path_output)
