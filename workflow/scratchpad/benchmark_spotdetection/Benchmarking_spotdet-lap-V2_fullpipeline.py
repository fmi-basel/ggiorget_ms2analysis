"""
This script performs the minimal example of my spot detection pipeline (simple linking) and compares the results with
ground-truth data. I sweep over several parameters to find the best settings for spot detection:
- pre-processing of the images: max-projection, max-projection + local background subtraction,
    max-projection + flat field correction, max-projection + flat field correction + local background subtraction
    max-projection + bleach correction, max-projection + bleach correction + local background subtraction
- diameter of the gaussian mask for local background subtraction
- diameter of the spots
- threshold for spot detection

Be aware that there is a filter step after spot detection for size and total intensity of the spots. These values are
not screened for and taken from an initial screening for best spot detection values. However, I do not expect them to
change
"""

# Let's start with loading all the packages
import os

import numpy as np
import pandas as pd
import trackpy as tp
from laptrack import LapTrack
from skimage import io

from utils import local_backgroundsubtraction

tp.quiet()


# I define some screening functions and only print out the number of spots detected (compared to ground truth)
def tracking_test(images, diameter, threshold, groundtruth, mask, mask_track):
    """
    Processes images with minimal example for spot detection pipeline.
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        diameter: (int) diameter of the spots to be detected
        threshold: (int) threshold for spot detection
        groundtruth: (pd.Dataframe) dataframe containing ground truth data
        mask: (nd.array) 3D array of the label image corresponding to the images (t, y, x)
        mask_track: (pd.Dataframe) dataframe containing the track information of the label image

    Returns:
        list of diameter, threshold, and number of false negatives, false positives, true positives and total number of
        spots, respectively for non-filtered and single positive filtered data

    """
    # spot detection
    df_spots = tp.batch(images, diameter=diameter, minmass=threshold)

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
    df_lap.rename(columns={'frame_y': 'frame'}, inplace=True)
    df_groundtruth.rename(columns={'track_id': 'track_id_cell'}, inplace=True)
    df_compare = pd.merge(df_lap.round({'y': 0, 'x': 0}), df_groundtruth, indicator=True,
             how='outer')
    df_compare['_merge'].value_counts()

    # small gaps allowed, keep max distance small
    max_distance = 20
    lt = LapTrack(track_cost_cutoff=max_distance ** 2,
                  gap_closing_max_frame_count=3,
                  gap_closing_cost_cutoff=max_distance ** 2,
                  splitting_cost_cutoff=max_distance ** 2,
                  merging_cost_cutoff=max_distance ** 2,
                  )

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
    df_lap['track_id_gapmin'] = df_lap.groupby(['track_id_cell', 'track_id']).ngroup()

    # exclude rows in which track_id_gap occurs only once
    df_lap = df_lap[df_lap['track_id_gapmin'].map(df_lap['track_id_gapmin'].value_counts()) > 1]

    # second LAP, try to link all spots in a cell, if a cell has more than 2 linkes traces, take the brighter one
    try:
        max_distance2 = 100
        lt2 = LapTrack(track_cost_cutoff=max_distance2 ** 2,
                       gap_closing_max_frame_count=df_lap['frame_y'].max(),
                       gap_closing_cost_cutoff=max_distance2 ** 2,
                       splitting_cost_cutoff=max_distance2 ** 2,
                       merging_cost_cutoff=max_distance2 ** 2,
                       )

        df_lap2 = []
        for cell in df_lap['track_id_cell'].dropna().unique():
            # select spots from one cell, artificially include rows for all frames, otherwise LAP gives error (empty rows)
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
        df_lap2['track_id_gapmax'] = df_lap2.groupby(['track_id_cell', 'track_id']).ngroup()

        # For every track, calculate the average intensity and length, if in a cell are multiple tracks, chose the longest one
        df_track_intensity = df_lap2.groupby('track_id_gapmax').agg(mean_intensity=('mass', 'mean'),
                                                                    count=('track_id_cell', 'size'),
                                                                    track_id_cell=('track_id_cell', 'unique'),
                                                                    sum_intensity=('mass', 'sum')).reset_index()
        df_track_intensity['track_id_cell'] = df_track_intensity['track_id_cell'].apply(lambda x: x[0])
        df_track_intensity = df_track_intensity.loc[df_track_intensity.groupby('track_id_cell')['sum_intensity'].idxmax()]

        df_lap2 = df_lap2.merge(df_track_intensity['track_id_gapmax'])

    except ValueError:
        # print('No spots survived first round of filtering')
        # follows idea: [diameter, threshold, FN, FP, TP, sumspots]
        return [diameter, threshold, len(groundtruth), 0, 0, 0]

    # round to integer values
    df_lap2 = df_lap2.round({'y': 0, 'x': 0})
    df_lap2.rename(columns={'frame_y': 'frame'}, inplace=True)

    # calculate false negatives, false positives and true positives
    FN = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_lap2[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    FP = len(pd.merge(df_lap2[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    TP = len(pd.merge(df_lap2[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))
    sumspots = len(df_lap2)

    return [diameter, threshold, FN, FP, TP, sumspots]


def sweep_tracking(images, list_diameter, list_threshold):
    """
    Wrapper function for trackpy_test to perform a parameter sweep over diameter and threshold.
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        list_diameter: (list) list of int values for diameter values (trackpy)
        list_threshold: (list) list of int values for threshold values (trackpy)

    Returns: pd.Dataframe containing the results of the parameter sweep
    """
    summary = []
    for thres in list_threshold:
        out1 = []
        for dia in list_diameter:
            out2 = tracking_test(images, diameter=dia, threshold=thres, groundtruth=df_groundtruth, mask=mask_image,
                                 mask_track=mask_tracks)
            out1.append(out2)
        summary.extend(out1)
    df_out = pd.DataFrame(summary, columns=['diameter', 'threshold', 'FN', 'FP', 'TP', 'sumspots'])
    return df_out


# ---- load data ----
# max projected images and corresponding label images
path_images = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230715_30s'
# path_images = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20230715_30s'
filename = '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_cutMAX.tiff'
images_maxproj = io.imread(os.path.join(path_images, filename))

mask_labelimagename = filename.replace('_cutMAX.tiff', '_label-image.tif')
mask_image = io.imread(os.path.join(path_images, mask_labelimagename))
mask_tracks = pd.read_csv(os.path.join(path_images, mask_labelimagename.replace('.tif', '_tracks.csv')))

# images for flat field correction
path_flatfield = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20230715/'
# path_flatfield = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20230715/'
flatfield_cy5_filename = 'AVG_20230715_A568_100000_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
darkimage_cy5_filename = 'AVG_20230715_Darkimage_1_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
flatfield_cy5_image = io.imread(os.path.join(path_flatfield, flatfield_cy5_filename))
dark_cy5_image = io.imread(os.path.join(path_flatfield, darkimage_cy5_filename))

# load groundtruth data, pre-process by removing spots that are not assigned to cells
path_groundtruth = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/groundtruth/'
# path_groundtruth = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/groundtruth/'
filename_groundtruth = filename.replace('MAX.tiff', 'tracks_groundtruth.csv')
df_groundtruth = pd.read_csv(os.path.join(path_groundtruth, filename_groundtruth))
t = df_groundtruth['frame'].astype(np.int64)
y = df_groundtruth['y'].astype(np.int64)
x = df_groundtruth['x'].astype(np.int64)
df_groundtruth['track_id'] = mask_image[t, y, x]
df_groundtruth = df_groundtruth[df_groundtruth.track_id != 0]
df_groundtruth = df_groundtruth.round(0)

# ---- parameter sweep ----
# -- Max-Projection --
df_maxproj = sweep_tracking(images_maxproj, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
df_maxproj['images'] = 'Maxprojection'

# -- Max-Projection + local background subtraction --
# additionally to detection parameter screen, screen over different pixel sizes for the local background subtraction
pixel_range = [3, 5, 7, 9]

df_maxsub = []
for pixel in pixel_range:
    images_maxsub = np.stack(
        [local_backgroundsubtraction(images_maxproj[i, ...], pixelsize=pixel) for i in range(images_maxproj.shape[0])],
        axis=0)

    df = sweep_tracking(images_maxsub, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
    df['images'] = f'Maxprojection_backgroundsub{pixel}'
    df_maxsub.append(df)
df_maxsub = pd.concat(df_maxsub)

"""
# -- Max-Projection + bleach correction --
# Bleach correction by simple ratio
I_mean = np.mean(images_maxproj, axis=(1, 2))
I_ratio = I_mean[0] / I_mean
I_ratio = I_ratio.reshape(-1, 1, 1)

images_ratio = I_ratio * images_maxproj

df_ratio = sweep_trackpy(images_ratio, list_diameter=[7, 9, 11], list_threshold=range(3000, 6000, 200))
df_ratio['images'] = 'Maxprojection_ratio'

# -- Max-Projection + bleach correction + local background subtraction --
# additionally to detection parameter screen, screen over different pixel sizes for the local background subtraction
pixel_range = [3, 5, 7, 9]

df_ratiosub = []
for pixel in pixel_range:
    images_ratiosub = np.stack(
        [local_backgroundsubtraction(images_ratio[i, ...], pixelsize=pixel) for i in range(images_ratio.shape[0])],
        axis=0)

    df = sweep_trackpy(images_ratiosub, list_diameter=[7, 9, 11], list_threshold=range(3000, 6000, 200))
    df['images'] = f'Maxprojection_ratio_backgroundsub{pixel}'
    df_ratiosub.append(df)
df_ratiosub = pd.concat(df_ratiosub)

# -- Max-Projection + flat field correction --
images_maxcorr = np.stack(
    [flatfieldcorrection(images_maxproj[i, ...], flatfield_cy5_image, dark_cy5_image) for i in
     range(images_maxproj.shape[0])],
    axis=0)

df_maxcorr = sweep_trackpy(images_maxcorr, list_diameter=[7, 9, 11], list_threshold=range(4000, 7000, 200))
df_maxcorr['images'] = 'Maxprojection_corr'

# -- Max-Projection + flat field correction + local background subtraction --
pixel_range = [3, 5, 7, 9]

df_maxcorrsub = []
for pixel in pixel_range:
    images_maxcorrsub = np.stack(
        [local_backgroundsubtraction(images_maxcorr[i, ...], pixelsize=pixel) for i in range(images_maxcorr.shape[0])],
        axis=0)

    df = sweep_trackpy(images_maxcorrsub, list_diameter=[7, 9, 11], list_threshold=range(4000, 7000, 200))
    df['images'] = f'Maxprojection_corr_backgroundsub{pixel}'
    df_maxcorrsub.append(df)
df_maxcorrsub = pd.concat(df_maxcorrsub)

, df_ratio, df_ratiosub, df_maxcorr, df_maxcorrsub
"""
# save summary
df_summary = pd.concat([df_maxproj, df_maxsub], ignore_index=True)
df_summary.to_csv(os.path.join(path_groundtruth, filename.replace('.tiff', '_summary_spotdet-lapV2_fullpipeline.csv')),
                  index=False)

print('Done :)')
