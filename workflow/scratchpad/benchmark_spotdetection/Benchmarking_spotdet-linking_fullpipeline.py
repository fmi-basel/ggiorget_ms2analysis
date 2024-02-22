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
from skimage import io

from utils import local_backgroundsubtraction, flatfieldcorrection

tp.quiet()


# I define some screening functions and only print out the number of spots detected (compared to ground truth)
def remove_singlepositives(x):
    """
    From a series calculate how long a state was present.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
    Returns:
         x_filtered: (pd.Dataframe) dataframe in which rows of single positive occurances are removed

    """
    x_filtered = x.copy()
    x_filtered['prev_frame'] = x_filtered['frame'] - x_filtered['frame'].shift(1)
    x_filtered['next_frame'] = x_filtered['frame'] - x_filtered['frame'].shift(-1)
    x_filtered['remove'] = (x_filtered['prev_frame'] != 1) & (x_filtered['next_frame'] != -1)
    x_filtered = x_filtered[~x_filtered['remove']]
    return x_filtered


def trackpy_test(images, diameter, threshold, groundtruth, mask):
    """
    Processes images with minimal example for spot detection pipeline.
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        diameter: (int) diameter of the spots to be detected
        threshold: (int) threshold for spot detection
        groundtruth: (pd.Dataframe) dataframe containing groundtruth data
        mask: (nd.array) 3D array of the labelimage corresponding to the images (t, y, x)

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

    # only keep the brightest spot
    df_spots = df_spots.groupby(['track_id', 'frame']).apply(lambda x: x.loc[x.mass.idxmax()]).reset_index(
        drop=True)

    # remove single positives in time course
    df_spots = df_spots.sort_values(['track_id', 'frame']).reset_index(drop=True)
    df_spots_filtered = df_spots.groupby('track_id').apply(lambda x: remove_singlepositives(x))

    # round to integer values
    df_spots = df_spots.round({'y': 0, 'x': 0})
    df_spots_filtered = df_spots_filtered.round({'y': 0, 'x': 0})

    # calculate false negatives, false positives and true positives
    FN = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_spots[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    FP = len(pd.merge(df_spots[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                      how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    TP = len(pd.merge(df_spots[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))
    sumspots = len(df_spots)

    FN_filtered = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_spots_filtered[['frame', 'y', 'x']], indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    FP_filtered = len(pd.merge(df_spots_filtered[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                               how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    TP_filtered = len(pd.merge(df_spots_filtered[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))
    sumspots_filtered = len(df_spots_filtered)

    return [diameter, threshold, FN, FP, TP, sumspots, FN_filtered, FP_filtered, TP_filtered, sumspots_filtered]


def sweep_trackpy(images, list_diameter, list_threshold):
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
            out2 = trackpy_test(images, diameter=dia, threshold=thres, groundtruth=df_groundtruth, mask=mask_image)
            out1.append(out2)
        summary.extend(out1)
    df_out = pd.DataFrame(summary, columns=['diameter', 'threshold', 'FN', 'FP', 'TP', 'sumspots', 'FN_filtered',
                                            'FP_filtered', 'TP_filtered', 'sumspots_filtered'])
    return df_out


# ---- load data ----
# max projected images and corresponding label images
path_images = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230719_30s'
filename = '20230719_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5E10_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_MAX.tiff'
images_maxproj = io.imread(os.path.join(path_images, filename))

mask_labelimagename = filename.replace('_MAX.tiff', '_label-image.tif')
mask_image = io.imread(os.path.join(path_images, mask_labelimagename))

# images for flat field correction
path_flatfield = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20230719/'
flatfield_cy5_filename = 'AVG_20230719_A568_100000_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
darkimage_cy5_filename = 'AVG_20230719_Darkimage_1_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
flatfield_cy5_image = io.imread(os.path.join(path_flatfield, flatfield_cy5_filename))
dark_cy5_image = io.imread(os.path.join(path_flatfield, darkimage_cy5_filename))

# load groundtruth data, pre-process by removing spots that are not assigned to cells
path_groundtruth = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/groundtruth/'
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
df_maxproj = sweep_trackpy(images_maxproj, list_diameter=[7, 9, 11], list_threshold=range(3000, 6000, 200))
df_maxproj['images'] = 'Maxprojection'

# -- Max-Projection + local background subtraction --
# additionally to detection parameter screen, screen over different pixel sizes for the local background subtraction
pixel_range = [3, 5, 7, 9]

df_maxsub = []
for pixel in pixel_range:
    images_maxsub = np.stack(
        [local_backgroundsubtraction(images_maxproj[i, ...], pixelsize=pixel) for i in range(images_maxproj.shape[0])],
        axis=0)

    df = sweep_trackpy(images_maxsub, list_diameter=[7, 9, 11], list_threshold=range(3000, 6000, 200))
    df['images'] = f'Maxprojection_backgroundsub{pixel}'
    df_maxsub.append(df)
df_maxsub = pd.concat(df_maxsub)

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

# save summary
df_summary = pd.concat([df_maxproj, df_maxsub, df_ratio, df_ratiosub, df_maxcorr, df_maxcorrsub], ignore_index=True)
df_summary.to_csv(os.path.join(path_groundtruth, filename.replace('.tiff', '_summary_spotdet-filter_fullpipeline.csv')),
                  index=False)

print('Done :)')
