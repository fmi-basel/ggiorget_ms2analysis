import argparse
import os
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import io
from skimage.feature import blob_log
from skimage.measure import regionprops
from tqdm import tqdm


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


def _bbox_to_slices(bbox, padding, shape):
    y_slice = slice(max(0, bbox[0] - padding[0]), min(shape[0], bbox[2] + padding[0]))
    x_slice = slice(max(0, bbox[1] - padding[1]), min(shape[1], bbox[3] + padding[1]))
    return y_slice, x_slice


def _crop_padded_roi_img(image, roi, segmentation, yx_sigma):
    y_slice, x_slice = _bbox_to_slices(
        roi.bbox,
        padding=(
            min(1, round(3 * yx_sigma)),
            min(1, round(3 * yx_sigma)),
        ),
        shape=segmentation.shape,
    )
    roi_img = image[y_slice, x_slice]
    return roi_img, x_slice, y_slice


def detect_spots_log(
        img_path: str,
        cell_seg_path: str,
        spotdiameter_min: float,
        spotdiameter_max: float,
        threshold: int
) -> pd.DataFrame:
    """
    Detect bright, diffraction limited spots.

    The Laplacian of Gaussian is used to detect diffraction limited spots,
    where the spot size is computed from the emission `wavelength`, `NA` and
    pixel `spacing`. This results in an over-detection of spots and only the
    ones with an intensity larger than `intensity_threshold` relative to
    their immediate neighborhood are kept.

    Parameters
    ----------
    img_path :
        Path to the raw image data.
    cell_seg_path :
        Path to the ROI segmentation image.
    spotdiameter_min :
        Minimal spot diameter in order of sigma
    spotdiameter_max :
        Maximal spot diameter in order of sigma
    threshold :
        Minimum spot intensity relative to the immediate background (ranges from 0-1)

    Returns
    -------
    pd.DataFrame with detected spots
    """
    image_proj = io.imread(img_path)
    image = np.stack(
        [local_backgroundsubtraction(image_proj[i, ...], pixelsize=5) for i in range(image_proj.shape[0])],
        axis=0)
    segmentation = io.imread(cell_seg_path).astype(np.uint16)
    spots = []
    for frame in tqdm(range(0, image.shape[0])):
        for roi in regionprops(segmentation[frame, ...]):
            roi_img, x_slice, y_slice = _crop_padded_roi_img(
                image=image[frame, ...],
                roi=roi,
                segmentation=segmentation[frame, ...],
                yx_sigma=spotdiameter_max
            )

            log_detections = blob_log(roi_img, min_sigma=spotdiameter_min, max_sigma=spotdiameter_max,
                                      threshold_rel=threshold)
            log_detections += np.array([y_slice.start, x_slice.start, 0])
            spots.extend(np.insert(log_detections, 0, frame, axis=1).tolist())

    # duplicates can occur if cells are too close/crops overlap. Simply remove them
    spots = pd.DataFrame(spots, columns=["frame", "y", "x", 'sigma']).drop_duplicates().reset_index(drop=True)

    return spots


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
    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    image_path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/processed/20230715/proj/20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_MAX.tiff'
    groundtruth_path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/data/groundtruth_tracks/20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv'
    # 1. ------- Data loading and Pre-processing--------
    path_images = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    path_mask = path_images.replace('proj', 'segmentation')
    filename_mask = filename.replace('_MAX.tiff', '_label-image.tiff')
    mask_image = io.imread(os.path.join(path_mask, filename_mask))

    df_groundtruth = pd.read_csv(groundtruth_path)
    df_groundtruth = df_groundtruth.rename(columns={'axis-0': 'frame', 'axis-1': 'y', 'axis-2': 'x'})

    # ---- sweep ----
    # define parameters to sweep
    spotdiameter = np.round(np.arange(0.1, 1, 0.01), decimals=2)
    #thresholds = np.round(np.arange(0.6, 0.9, 0.1), decimals=1)
    thresholds = 0.1
    spotdiameter_combinations = [(x, y) for x, y in combinations_with_replacement(spotdiameter, 2) if x < y]
    parameter_list = list(product(spotdiameter_combinations, thresholds))

    # data structure to store the results
    path_output_spots = os.path.join(path_output, 'log', 'spots')
    filename_spots = filename.replace('_MAX.tiff', '_spots.csv')
    path_output_comparison = os.path.join(path_output, 'log', 'comparison')

    # sweep
    sweep_results = []
    for parameter in parameter_list:
        spotdiameter, threshold = parameter
        spotdiameter_min, spotdiameter_max = spotdiameter

        threshold = 0.001
        spotdiameter_min = 0.1
        spotdiameter_max = 0.2

        df_spots = detect_spots_log(
            img_path=image_path,
            cell_seg_path=os.path.join(path_mask, filename_mask),
            spotdiameter_min=spotdiameter_min,
            spotdiameter_max=spotdiameter_min,
            threshold=threshold
        )

        path_output_spotsweep = os.path.join(path_output_spots,
                                             f'{spotdiameter_min}_{spotdiameter_max}_{threshold}')
        if not os.path.exists(path_output_spotsweep):
            # Create a new directory because it does not exist
            os.makedirs(path_output_spotsweep)
        df_spots.to_csv(os.path.join(path_output_spotsweep, filename_spots), index=False)

        # assign spots to cells
        t = df_spots['frame'].astype(np.int64)
        y = df_spots['y'].astype(np.int64)
        x = df_spots['x'].astype(np.int64)
        df_spots['track_id'] = mask_image[t, y, x]
        # Let's get rid of spots which are not assigned to cells
        df_spots = df_spots[df_spots.track_id != 0]

        # compare to ground-truth data
        comparing_df = []
        for frame in np.unique(pd.concat([df_spots['frame'], df_groundtruth['frame']])):
            # compare frame by frame
            match_df = linear_sum_colocalisation(df_spots[df_spots['frame'] == frame],
                                                 df_groundtruth[df_groundtruth['frame'] == frame], 4)
            # instead of the index, we want to know if a spot was correctly detected or not
            match_df['spot_detected'] = ~match_df['index_spot'].isna()
            match_df['groundtruth'] = ~match_df['index_groundtruth'].isna()
            # add frame information
            match_df['frame'] = frame
            comparing_df.append(match_df)
        comparing_df = pd.concat(comparing_df)
        # add more useful information
        comparing_df['threshold'] = threshold
        comparing_df['spotdiameter_min'] = spotdiameter_min
        comparing_df['spotdiameter_max'] = spotdiameter_max
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
