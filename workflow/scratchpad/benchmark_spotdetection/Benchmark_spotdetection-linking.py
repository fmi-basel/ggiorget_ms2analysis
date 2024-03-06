import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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


def spotdetection(image_path, mask_image_path, path_output, spotdiameter, threshold, size_threshold, mass_threshold):
    tp.quiet()
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(image_path)[1]

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

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
    df_spots = tp.batch(images_sub, diameter=spotdiameter, minmass=threshold, processes=10)

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
    threshold_size = size_threshold  # default 1.19 for JF646, 1.29 for JF549
    threshold_mass = mass_threshold  # default 21000 for JF646, 30000 for JF549

    # If spots are detected, create a plot to visualize the thresholds
    try:
        # Create the plot and save it
        plt.scatter(df_spots['mass'], df_spots['size'])
        plt.hlines(y=threshold_size, xmin=min(df_spots['mass']), xmax=max(df_spots['mass']), colors='r',
                   linestyles='--')
        plt.vlines(x=threshold_mass, ymin=min(df_spots['size']), ymax=max(df_spots['size']), colors='r',
                   linestyles='--')
        plt.ylabel('Size')
        plt.xlabel('Mass')
        plt.title(images_filename, fontsize=5)
        plt.savefig(os.path.join(path_output, images_filename.replace('_MAX.tiff', '_Spot-Filter.pdf')))
        # plt.show()
        plt.close('all')
    except ValueError:
        print('No spots detected, skipping plot')

    # Remove the camera error spots by thresholding
    df_spots = df_spots[df_spots['mass'] <= threshold_mass]
    df_spots = df_spots[df_spots['size'] >= threshold_size]

    # save spots
    df_spots.to_csv(os.path.join(path_output, images_filename.replace('_MAX.tiff', '_spots.csv')), index=False)
    print('Done :)')


def linear_sum_colocalisation(spot_coord_1: pd.DataFrame, spot_coord_2: pd.DataFrame, cutoff_dist: int):
    """
    Compare two dataframes with spot coordinates and indicate if spot is present in both dataframes with a certain
    cutoff distance.

    Parameters
    ----------
    spot_coord_1 :
        Dataframe with first set of spot coordinates

    spot_coord_2 :
        Dataframe with second set of spot coordinates

    cutoff_dist :
        cutoff distance in pixels (sum over all axis)

    Returns
    -------
    input dataframes with additional column indicating co-localisation and one additional dataframe with matched spots
    """
    spot_coord_1_indexed = spot_coord_1.reset_index()
    spot_coord_1_indexed.rename(columns={'index': 'index_spot'}, inplace=True)
    spot_coord_2_indexed = spot_coord_2.reset_index()
    spot_coord_2_indexed.rename(columns={'index': 'index_groundtruth'}, inplace=True)

    spot_coord_1_values = spot_coord_1_indexed[['y', 'x']].dropna().values
    spot_coord_2_values = spot_coord_2_indexed[['y', 'x']].dropna().values

    # if one of the dataframes is empty, cdist can cause an error, if that's the case, return empty indexes
    try:
        # calculate the distance matrix and find the optimal assignment
        global_distances = cdist(spot_coord_1_values, spot_coord_2_values, 'euclidean')
        spot_coord_1_ind, spot_coord_2_ind = linear_sum_assignment(global_distances)

        # use threshold on distance
        assigned_distance = spot_coord_1_values[spot_coord_1_ind] - spot_coord_2_values[spot_coord_2_ind]
        distance_id = np.sum(np.abs(assigned_distance), axis=1)

        spot_coord_1_ind = spot_coord_1_ind[distance_id < cutoff_dist]
        spot_coord_2_ind = spot_coord_2_ind[distance_id < cutoff_dist]
    except ValueError:
        spot_coord_1_ind = []
        spot_coord_2_ind = []

    # create a dataframe only containing the matches spots, where one row corresponds to a match
    match_index_df = pd.DataFrame({'match_spot': spot_coord_1_ind, 'match_groundtruth': spot_coord_2_ind})

    match_df = pd.merge(match_index_df, spot_coord_1_indexed, left_on='match_spot', right_index=True, how='outer')
    match_df = pd.merge(match_df, spot_coord_2_indexed, left_on='match_groundtruth', right_index=True, how='outer')
    match_df = match_df[['index_spot', 'index_groundtruth']].reset_index(drop=True)

    return match_df


def main(image_path, groundtruth_path, path_output):
    # load data
    path_images = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    path_mask = path_images.replace('proj', 'segmentation')
    filename_mask = filename.replace('_MAX.tiff', '_label-image.tiff')

    df_groundtruth = pd.read_csv(groundtruth_path)
    df_groundtruth = df_groundtruth.rename(columns={'axis-0': 'frame', 'axis-1': 'y', 'axis-2': 'x'})

    path_output_spots = os.path.join(path_output, 'spots')
    filename_spots = filename.replace('_MAX.tiff', '_spots.csv')
    path_output_comparison = os.path.join(path_output, 'comparison')

    # define parameters to sweep
    param_5 = [5, 1, 20000]
    threshold_5 = range(1000, 3201, 200)
    list_5 = [[*param_5, threshold] for threshold in threshold_5]
    param_7 = [7, 1.19, 30000]
    threshold_7 = range(2200, 4201, 200)
    list_7 = [[*param_7, threshold] for threshold in threshold_7]
    param_9 = [9, 1.29, 40000]
    threshold_9 = range(3800, 5801, 200)
    list_9 = [[*param_9, threshold] for threshold in threshold_9]
    param_11 = [11, 1.49, 50000]
    threshold_11 = range(6000, 8000, 200)
    list_11 = [[*param_11, threshold] for threshold in threshold_11]
    param_list = list_5+list_7+list_9+list_11

    sweep_results = []
    for combination in param_list:
        spotdetection(image_path=image_path,
                      mask_image_path=os.path.join(path_mask, filename_mask),
                      path_output=os.path.join(path_output_spots, f'{combination[0]}_{combination[3]}'),
                      spotdiameter=combination[0], threshold=combination[3], size_threshold=combination[1],
                      mass_threshold=combination[2])

        # load spot data
        df_spots = pd.read_csv(os.path.join(path_output_spots, f'{combination[0]}_{combination[3]}', filename_spots))
        # spot linking
        # from linking only the removal of double spots is important
        df_spots = df_spots.groupby(['track_id', 'frame']).apply(lambda df: df.loc[df.mass.idxmax()]).reset_index(drop=True)

        # compare to ground-truth data
        comparing_df = []
        for frame in np.unique(pd.concat([df_spots['frame'], df_groundtruth['frame']])):
            # compare frame by frame
            match_df = linear_sum_colocalisation(df_spots[df_spots['frame'] == frame],
                                                 df_groundtruth[df_groundtruth['frame'] == frame], 2)
            # instead of the index, we want to know if a spot was correctly detected or not
            match_df['spot_detected'] = ~match_df['index_spot'].isna()
            match_df['groundtruth'] = ~match_df['index_groundtruth'].isna()
            # add frame information
            match_df['frame'] = frame
            comparing_df.append(match_df)
        comparing_df = pd.concat(comparing_df)
        # add more useful information
        comparing_df['spotdiameter'] = combination[0]
        comparing_df['threshold'] = combination[3]
        comparing_df['filename'] = filename
        sweep_results.append(comparing_df)
    sweep_results = pd.concat(sweep_results)

    # save results
    if not os.path.exists(path_output_comparison):
        # Create a new directory because it does not exist
        os.makedirs(path_output_comparison)

    sweep_results.to_csv(os.path.join(path_output_comparison, filename.replace('_MAX.tiff', '_sweep.csv')),
                         index=False)


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
        "-g",
        "--path_groundtruth",
        type=str,
        default=None,
        required=True,
        help="groundtruth data belonging to input movie, requires absolute path",
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

    main(image_path=args.path_input, groundtruth_path=args.path_groundtruth, path_output=args.path_output)
