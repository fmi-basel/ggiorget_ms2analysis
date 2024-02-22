"""
This script performs spot detection and compares the spots with ground-truth data. Two different spot detection
algorithms (trackpy and hmax) are performed with differently pre-processed images and several parameters sweeped.
- pre-processing of the images: max-projection, max-projection + local background subtraction,
    max-projection + flat field correction, max-projection + flat field correction + local background subtraction
    max-projection + bleach correction, max-projection + bleach correction + local background subtraction
- diameter of the gaussian mask for local background subtraction
- diameter of the spots
- threshold for spot detection
As output, a summary is saved of True-positives, False-positives and False-negatives.
"""

# Let's start with loading all the packages
import os

import numpy as np
import pandas as pd
import trackpy as tp
from skimage import io

from utils import local_backgroundsubtraction, flatfieldcorrection, hmax_detection

tp.quiet()


# I define some screening functions and only print out the number of spots detected (compared to
# groundtruth)
# Trackpy Approach
def trackpy_test(images, diameter, threshold, groundtruth):
    """
    Spot detection using the trackpy package
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        diameter: (int) diameter of the spots to be detected
        threshold: (int) threshold for spot detection (min_mass)
        groundtruth: (pd.Dataframe) dataframe containing ground truth data

    Returns:
        list of diameter, threshold, and number of false negatives, false positives, true positives and total number of
        spots
    """
    # spot detection
    df_spots = tp.batch(images, diameter=diameter, minmass=threshold)
    # round position to integer, sub pixel localization is not relevant for this analysis
    df_spots_round = df_spots.round(0)
    # compare with ground truth
    falseneg = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_spots_round[['frame', 'y', 'x']], indicator=True,
                            how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    falsepos = len(pd.merge(df_spots_round[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                            how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    truepos = len(pd.merge(df_spots_round[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))
    sumspots = len(df_spots)
    return [diameter, threshold, falseneg, falsepos, truepos, sumspots]


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
            out2 = trackpy_test(images, diameter=dia, threshold=thres, groundtruth=df_groundtruth)
            out1.append(out2)
        summary.extend(out1)
    df_out = pd.DataFrame(summary, columns=['diameter', 'threshold', 'FN', 'FP', 'TP', 'Sumspots'])
    return df_out


# hmax approach
def hmax_test(images, sd, groundtruth):
    """
    Spot detection using hmax
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        sd: (int) sd threshold for detection
        groundtruth: (pd.Dataframe) dataframe containing ground truth data

    Returns:
        list of sd threshold values and number of false negatives, false positives, true positives and total number of
        spots
    """
    out = []
    for frame in range(images.shape[0]):
        df = hmax_detection(images, frame=frame, sd=sd)
        out.append(df)
    df_spots = pd.concat(out)
    df_spots_round = df_spots.round(0)
    falseneg = len(pd.merge(groundtruth[['frame', 'y', 'x']], df_spots_round[['frame', 'y', 'x']], indicator=True,
                            how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    falsepos = len(pd.merge(df_spots_round[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']], indicator=True,
                            how='outer').query('_merge=="left_only"').drop('_merge', axis=1))
    truepos = len(pd.merge(df_spots_round[['frame', 'y', 'x']], groundtruth[['frame', 'y', 'x']]))
    sumspots = len(df_spots)
    return [sd, falseneg, falsepos, truepos, sumspots]


def sweep_hmax(images, sd_list):
    """
    Wrapper function for hmax_test to perform a parameter sweep over sd values
    Args:
        images: (nd.array) 3D array of an image series (t, y, x)
        sd_list: (list) list of int values for sd values (hmax)

    Returns: pd.Dataframe containing the results of the parameter sweep
    """
    summary = []
    for sd in sd_list:
        out = hmax_test(images, sd, groundtruth=df_groundtruth)
        summary.append(out)
    df_out = pd.DataFrame(summary, columns=['sd_threshold', 'FN', 'FP', 'TP', 'Sumspots'])
    return df_out


# load data
path_images = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230720_30s'
filename = '20230720_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5F11_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s5_MAX.tiff'
images_maxproj = io.imread(os.path.join(path_images, filename))

path_flatfield = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20230720/'
flatfield_cy5_filename = 'AVG_20230720_A568_100000_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
darkimage_cy5_filename = 'AVG_20230720_Darkimage_1_FullseqTIRF-mCherry-GFPCy5WithSMB.tif'
flatfield_cy5_image = io.imread(os.path.join(path_flatfield, flatfield_cy5_filename))
dark_cy5_image = io.imread(os.path.join(path_flatfield, darkimage_cy5_filename))

path_groundtruth = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/groundtruth/'
filename_groundtruth = filename.replace('_MAX.tiff', '_tracks_groundtruth.csv')
df_groundtruth = pd.read_csv(os.path.join(path_groundtruth, filename_groundtruth))
df_groundtruth = df_groundtruth.round(0)

# ---- Sweep ----
# -- Max-Projection --
df_maxproj = sweep_trackpy(images_maxproj, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
df_maxproj['images'] = 'Maxprojection'

# -- Max-Projection + local background subtraction --
# additionally to detection parameter screen, screen over different pixel sizes for the local background subtraction
pixel_range = [3, 5, 7, 9]

df_maxsub = []
for pixel in pixel_range:
    images_maxsub = np.stack(
        [local_backgroundsubtraction(images_maxproj[i, ...], pixelsize=pixel) for i in range(images_maxproj.shape[0])],
        axis=0)

    df = sweep_trackpy(images_maxsub, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
    df['images'] = f'Maxprojection_backgroundsub{pixel}'
    df_maxsub.append(df)
df_maxsub = pd.concat(df_maxsub)

# -- Max-Projection + bleach correction --
# Bleach correction by simple ratio
I_mean = np.mean(images_maxproj, axis=(1, 2))
I_ratio = I_mean[0] / I_mean
I_ratio = I_ratio.reshape(-1, 1, 1)
images_ratio = I_ratio * images_maxproj

# sweep over different detection parameters
df_ratio = sweep_trackpy(images_ratio, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
df_ratio['images'] = 'Maxprojection_ratio'

# -- Max-Projection + bleach correction + local background subtraction --
# additionally to detection parameter screen, screen over different pixel sizes for the local background subtraction
pixel_range = [3, 5, 7, 9]

df_ratiosub = []
for pixel in pixel_range:
    images_ratiosub = np.stack(
        [local_backgroundsubtraction(images_ratio[i, ...], pixelsize=pixel) for i in range(images_ratio.shape[0])],
        axis=0)

    df = sweep_trackpy(images_ratiosub, list_diameter=[7, 9, 11], list_threshold=range(4000, 6000, 200))
    df['images'] = f'Maxprojection_ratio_backgroundsub{pixel}'
    df_ratiosub.append(df)
df_ratiosub = pd.concat(df_ratiosub)

# -- Max-Projection + flat field correction --
images_maxcorr = np.stack(
    [flatfieldcorrection(images_maxproj[i, ...], flatfield_cy5_image, dark_cy5_image) for i in
     range(images_maxproj.shape[0])],
    axis=0)

df_maxcorr = sweep_trackpy(images_maxcorr, list_diameter=[7, 9, 11], list_threshold=range(5000, 7000, 200))
df_maxcorr['images'] = 'Maxprojection_corr'

# -- Max-Projection + flat field correction + local background subtraction --
pixel_range = [3, 5, 7, 9]

df_maxcorrsub = []
for pixel in pixel_range:
    images_maxcorrsub = np.stack(
        [local_backgroundsubtraction(images_maxcorr[i, ...], pixelsize=pixel) for i in range(images_maxcorr.shape[0])],
        axis=0)

    df = sweep_trackpy(images_maxcorrsub, list_diameter=[7, 9, 11], list_threshold=range(5000, 7000, 200))
    df['images'] = f'Maxprojection_corr_backgroundsub{pixel}'
    df_maxcorrsub.append(df)
df_maxcorrsub = pd.concat(df_maxcorrsub)

# save summary
df_summary = pd.concat([df_maxproj, df_maxsub, df_ratio, df_ratiosub, df_maxcorr, df_maxcorrsub], ignore_index=True)
df_summary.to_csv(os.path.join(path_groundtruth, filename.replace('.tiff', '_summary_spotdetec_trackpy_sub.csv')),
                  index=False)

"""
# hmax
# Max projection
df_maxproj = sweep_hmax(images_maxproj, sd_list=range(200, 300, 10))
df_maxproj['images'] = 'Maxprojection'

# SD projection
df_sd = sweep_hmax(images_sd, sd_list=range(40, 100, 5))
df_sd['images'] = 'SDprojection'

# Max projection + local Background subtraction
df_maxsub = sweep_hmax(images_maxsub, sd_list=range(250, 350, 10))
df_maxsub['images'] = 'Maxprojection_backgroundsub'

# Max projection + flat-field correction
df_maxcorr = sweep_hmax(images_maxcorr, sd_list=range(300, 400, 10))
df_maxcorr['images'] = 'Maxprojection_corr'

# Max projection + flat-field correction + local Background subtraction
df_maxcorrsub = sweep_hmax(images_maxcorrsub, sd_list=range(300, 400, 10))
df_maxcorrsub['images'] = 'Maxprojection_corr_backgroundsub'

# save summary
df_summary = pd.concat([df_maxproj, df_sd, df_maxsub, df_maxcorr, df_maxcorrsub], ignore_index=True)
df_summary.to_csv(os.path.join(path_groundtruth, filename.replace('.tiff', '_summary_spotdetec_hmax.csv')),
                  index=False)
"""
print('Done :)')
