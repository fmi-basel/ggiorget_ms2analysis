"""
This script encodes an objective function coding a minimal example of my spot detection pipeline using lap tracker to
link called spots and filter them based on temporal and spatial position. the pipeline contains following parameters
that are optimised:
- spot diameter
- threshold for spot detection
- step 1 max distance between spots
- step 1 max time gab between spots
- step 2 max distance between spots
- step 2 max time gab between spots

Be aware that there is a filter step after spot detection for size and total intensity of the spots. These values are
not screened for and taken from an initial screening for best spot detection values. However, I do not expect them to
change.
"""

# Let's start with loading all the packages
import os

import numpy as np
import pandas as pd
import trackpy as tp
from laptrack import LapTrack
from scipy.optimize import minimize
from skimage import io

from utils import local_backgroundsubtraction

tp.quiet()


# I define an objective function, which takes the parameters as input and returns the (1-F1 score) as output
def tracking_test(params):
    """
    Processes images with minimal example for spot detection pipeline.
    Args:
        params: (list) list of parameters to be optimised:
            diameter: (int, odd number) diameter of the spots to be detected
            threshold: (int) threshold for spot detection
            max_distance_step1: (int) maximal distance between two spots in pixel
            gabclosing_step1: (int) maximal time gab between two spots in frames
            max_distance_step2: (int) maximal distance between two spots in pixel
            gabclosing_step2: (int) maximal time gab between two spots in frames

    Returns:
        1 - F1 score: (float) 1 - F1 score of the spot detection pipeline
    """
    print(params)
    # unpack params, name images and pre-processing/background subtraction
    rounded_params = np.round(params).astype(int)
    diameter, threshold, max_distance_step1, gabclosing_step1, max_distance_step2, gabclosing_step2 = rounded_params

    groundtruth = df_groundtruth
    mask = mask_images
    mask_track = mask_tracks

    images = np.stack(
        [local_backgroundsubtraction(images_maxproj[i, ...], pixelsize=5) for i in range(images_maxproj.shape[0])],
        axis=0)

    print('rounding, background sub worked')
    print(rounded_params)
    # spot detection
    df_spots = tp.batch(images, diameter=diameter, minmass=threshold)
    print(params)
    print(rounded_params)

    # remove spots that are not assigned to cells (since edge removed)
    t = df_spots['frame'].astype(np.int64)
    y = df_spots['y'].astype(np.int64)
    x = df_spots['x'].astype(np.int64)
    df_spots['track_id'] = mask[t, y, x]
    df_spots = df_spots[df_spots.track_id != 0]

    # remove spots that are too small or too bright, values are taken from an initial screening for best spot detection
    # values
    threshold_size = 1.29
    threshold_mass = 30000
    df_spots = df_spots[df_spots['mass'] <= threshold_mass]
    df_spots = df_spots[df_spots['size'] >= threshold_size]

    # combine with mask info and calculate the relative position to the cells centroid
    df_spots = df_spots.merge(mask_track, on=['track_id', 'frame'])
    df_spots['y_rel'] = df_spots['y'] - df_spots['centroid-y']
    df_spots['x_rel'] = df_spots['x'] - df_spots['centroid-x']
    # rename track_id to track_id_cell, since it will be overwritten by LAP
    df_spots.rename(columns={'track_id': 'track_id_cell'}, inplace=True)

    # LAP tracking step 1
    # small gaps allowed, keep max distance small
    max_distance = max_distance_step1
    gabclosing = gabclosing_step1
    lt = LapTrack(track_cost_cutoff=max_distance ** 2,
                  gap_closing_max_frame_count=gabclosing,
                  gap_closing_cost_cutoff=max_distance ** 2,
                  splitting_cost_cutoff=max_distance ** 2,
                  merging_cost_cutoff=max_distance ** 2,
                  )

    # do tracking cell by cell
    df_lap = []
    for cell in df_spots['track_id_cell'].dropna().unique():
        # select spots from one cell, artificially include rows for all frames, otherwise LAP gives error (empty rows)
        df_cell = df_spots[df_spots['track_id_cell'] == cell]
        frames = pd.Series(np.arange(df_cell['frame'].max() + 1), name='frame')
        df_cell = pd.merge(frames, df_cell, how='left')
        # predict the tracks
        df_lapcell, _, _ = lt.predict_dataframe(df_cell, ["y", "x"], frame_col='frame',
                                                only_coordinate_cols=False)
        # remove artificially included rows
        df_lapcell = df_lapcell.dropna()
        # save the results
        df_lap.append(df_lapcell)
    df_lap = pd.concat(df_lap).reset_index(drop=True)
    # create unique id for each track (max gap size)
    df_lap['track_id_step1'] = df_lap.groupby(['track_id_cell', 'track_id']).ngroup()

    # exclude rows in which track_id_gap occurs only once
    df_lap = df_lap[df_lap['track_id_step1'].map(df_lap['track_id_step1'].value_counts()) > 1]

    # second LAP, try to link all spots in a cell, if tracks of 2 are not linked in, remove them
    # to deal with scenarios where no spot have been found in step 1, try this part first, otherwise return step1 df
    try:
        max_distance2 = max_distance_step2
        gabclosing2 = gabclosing_step2
        lt2 = LapTrack(track_cost_cutoff=max_distance2 ** 2,
                       gap_closing_max_frame_count=gabclosing2,
                       gap_closing_cost_cutoff=max_distance2 ** 2,
                       splitting_cost_cutoff=max_distance2 ** 2,
                       merging_cost_cutoff=max_distance2 ** 2,
                       )

        # do tracking cell by cell
        df_lap2 = []
        for cell in df_lap['track_id_cell'].dropna().unique():
            # select spots from one cell, artificially include rows for all frames, otherwise LAP gives error
            df_cell = df_lap[df_lap['track_id_cell'] == cell]
            frames = pd.Series(np.arange(df_cell['frame_y'].max() + 1), name='frame_y')
            df_cell = pd.merge(frames, df_cell, how='left')
            # predict the tracks
            df_lapcell, _, _ = lt2.predict_dataframe(df_cell, ["y_rel", "x_rel"], frame_col='frame_y',
                                                     only_coordinate_cols=False)
            # remove artificially included rows
            df_lapcell = df_lapcell.dropna()
            # save the results
            df_lap2.append(df_lapcell)
        df_lap2 = pd.concat(df_lap2).reset_index(drop=True)
        # create unique id for each track (max gap size)
        df_lap2['track_id_step2'] = df_lap2.groupby(['track_id_cell', 'track_id']).ngroup()

        # exclude rows in which track_id_step2 occurs only twice
        df_lap2 = df_lap2[df_lap2['track_id_step2'].map(df_lap2['track_id_step2'].value_counts()) > 2]

    except ValueError:
        # print('No spots survived first round of filtering')
        df_lap2 = df_lap

    # Compare to ground truth data
    # round to integer values
    df_lap2 = df_lap2.round({'y': 0, 'x': 0})
    df_lap2.rename(columns={'frame_y': 'frame'}, inplace=True)

    # calculate false negatives, false positives and true positives
    FN = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_lap2[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    FP = len(pd.merge(df_lap2[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    TP = len(pd.merge(df_lap2[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))

    F1score = (2 * TP) / (TP + FP + FN)
    score = 1 - F1score

    return score


# ---- load data ----
# Load and combine three movies
# Maxprojections
path_5G7 = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230715_30s'
# path_5G7 = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20230715_30s'
filename_5G7 = '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_cutMAX.tiff'
path_5E10 = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230719_30s'
# path_5E10 = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20230719_30s'
filename_5E10 = '20230719_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5E10_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_MAX.tiff'
path_5F11 = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230720_30s'
# path_5F11 = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20230720_30s'
filename_5F11 = '20230720_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5F11_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s5_MAX.tiff'

filenames = [
    (path_5G7, filename_5G7),
    (path_5E10, filename_5E10),
    (path_5F11, filename_5F11)]

images_maxproj = []
for path, filename in filenames:
    image = io.imread(os.path.join(path, filename))
    images_maxproj.append(image)
images_maxproj = np.concatenate(images_maxproj)

# labelimage tracks, create unique id
mask_tracks_5G7 = pd.read_csv(os.path.join(path_5G7, filename_5G7.replace('cutMAX.tiff', 'label-image_tracks.csv')))
mask_tracks_5G7 = mask_tracks_5G7[mask_tracks_5G7['frame'] <= 600]
mask_tracks_5G7['clone'] = '5G7'
mask_tracks_5E10 = pd.read_csv(os.path.join(path_5E10, filename_5E10.replace('MAX.tiff', 'label-image_tracks.csv')))
mask_tracks_5E10['frame'] = mask_tracks_5E10['frame'] + 601
mask_tracks_5E10['clone'] = '5E10'
mask_tracks_5F11 = pd.read_csv(os.path.join(path_5F11, filename_5F11.replace('MAX.tiff', 'label-image_tracks.csv')))
mask_tracks_5F11['frame'] = mask_tracks_5F11['frame'] + (601 * 2)
mask_tracks_5F11['clone'] = '5F11'
mask_tracks = pd.concat([mask_tracks_5G7, mask_tracks_5E10, mask_tracks_5F11])
mask_tracks['unique_id'] = mask_tracks.groupby(['clone', 'track_id']).ngroup()

# labelimage, relabel based on unique id
filenames = [
    (path_5G7, filename_5G7.replace('cutMAX.tiff', 'label-image.tif')),
    (path_5E10, filename_5E10.replace('MAX.tiff', 'label-image.tif')),
    (path_5F11, filename_5F11.replace('MAX.tiff', 'label-image.tif'))]

masks = []
for path, filename in filenames:
    image = io.imread(os.path.join(path, filename))[0:601, :, :]
    masks.append(image)
masks = np.concatenate(masks)

# create the new label image time series (tracked)
mask_images = np.zeros_like(masks)
for i, row in mask_tracks.iterrows():
    frame = int(row["frame"])
    inds = masks[frame] == row["track_id"]
    mask_images[frame][inds] = int(row["unique_id"]) + 1

# load groundtruth data
path_groundtruth = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/groundtruth/'
# path_groundtruth = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/groundtruth/'
filenames = [
    (filename_5G7.replace('MAX.tiff', 'tracks_groundtruth.csv')),
    (filename_5E10.replace('MAX.tiff', 'tracks_groundtruth.csv')),
    (filename_5F11.replace('MAX.tiff', 'tracks_groundtruth.csv'))]

df_groundtruth = []
for id, filename in enumerate(filenames):
    df = pd.read_csv(os.path.join(path_groundtruth, filename))
    df['frame'] = df['frame'] + (601 * id)
    df_groundtruth.append(df)
df_groundtruth = pd.concat(df_groundtruth)
t = df_groundtruth['frame'].astype(np.int64)
y = df_groundtruth['y'].astype(np.int64)
x = df_groundtruth['x'].astype(np.int64)
df_groundtruth['track_id'] = mask_images[t, y, x]
df_groundtruth = df_groundtruth[df_groundtruth.track_id != 0]
df_groundtruth = df_groundtruth.round(0)

# ---- parameter sweep ----
# Initial guess for the parameters (you can choose any starting point)
# [diameter, threshold, , gabclosing_step1, max_distance_step2, gabclosing_step2]
initial_guess = [11, 4600, 5, 100, 100, 600]

# Define the bounds for each parameter
param_bounds = [(7, 12), (3000, 6001), (0, 201), (0, 602), (0, 201), (0, 602)]

# Define constraints (diameter needs to be odd)
def integer_constraint(params):
    return [param - round(param) for param in params]
def odd_integer_constraint(params):
    x1 = params[0]
    return x1 % 2

constraints = (
    {'type': 'eq', 'fun': integer_constraint},
    {'type': 'eq', 'fun': odd_integer_constraint},  # Odd first parameter
    {'type': 'eq', 'fun': lambda params: np.logical_xor((params[1] - 4600) % 100, 1)},  # second parameter of a step size 100
               )

# Perform the optimization
result = minimize(tracking_test, initial_guess, method='SLSQP', bounds=param_bounds, constraints=constraints, options={'disp': True})

# Extract the optimal parameters and the minimum value of the function
optimal_params = result.x
min_function_value = result.fun

# Check if the constraints are satisfied
constraints_satisfied = all(constraint['fun'](optimal_params) == 0 for constraint in constraints)

print("Optimal Parameters:", optimal_params)
print("Minimum Function Value:", min_function_value)
print("Constraints Satisfied:", constraints_satisfied)

# df_summary = pd.DataFrame(data=[optimal_params, min_function_value])
# save summary
# df_summary.to_csv(os.path.join(path_groundtruth, 'test.csv'), index=False)

print('Done :)')
