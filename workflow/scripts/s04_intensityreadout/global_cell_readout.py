import numpy as np
import pandas as pd
from scipy import stats
from skimage import measure


def wholecell_readout_timeseries(images, mask, coordinates='', radius=9, excludearea=False):
    """
    Intensity read-out for a whole cell (timeseries).
    Given a movie and a corresponding mask, this function reads-out the intensity of the masked movie. If excludearea
    is active, before reading out the intensity, the area around the coordinates (using a radius) is
    excluded from the read-out.

    Args:
         images: (ndarray) movie from which the intensity is read out.
         mask: (ndarray) mask movie from images, defines region to read-out.
         coordinates (list) list with t,y,x coordinates of spots to exclude from read-out
         radius: (int) radius of ROI to exclude in pixel
         excludearea: (bool) if True, the area around the coordinates is excluded from the read-out

    Returns:
         df_intensity: (pd.dataframe) dataframe containing t,y,x coordinates and intensity values (Mean, SD, Median,
         MAD)
    """
    df_intensity_complete = []
    for timepoint, (timepoint_image, timepoint_mask) in enumerate(zip(images, mask)):
        df_intensity = wholecell_readout_singleframe(timepoint_image, timepoint_mask, coordinates, radius, excludearea)
        df_intensity['frame'] = timepoint
        df_intensity_complete.append(df_intensity)
    df_intensity_complete = pd.concat(df_intensity_complete)

    return df_intensity_complete


def wholecell_readout_singleframe(images, mask, coordinates='', radius=9, excludearea=False):
    """
    Intensity read-out for a whole cell (single image).
    Given an image and a corresponding mask, this function reads-out the intensity of the masked image. If excludearea
    is active, before reading out the intensity, the area around the coordinates (using a radius) is
    excluded from the read-out.

    Args:
         images: (ndarray) movie from which the intensity is read out.
         mask: (ndarray) mask movie from images, defines region to read-out.
         coordinates (list) list with t,y,x coordinates of spots to exclude from read-out
         radius: (int) radius of ROI to exclude in pixel
         excludearea: (bool) if True, the area around the coordinates is excluded from the read-out

    Returns:
         df_intensity: (pd.dataframe) dataframe containing t,y,x coordinates and intensity values (Mean, SD, Median,
         MAD)
    """

    df_intensity_complete = pd.DataFrame(measure.regionprops_table(mask, images, properties=('label', 'area',),
                                                                   extra_properties=(
                                                                       mean, sd, median, mad, integrated_intensity,)))

    if excludearea:
        mask_exclude = mask.copy()
        for row in np.arange(0, len(coordinates)):
            y = round(coordinates[row][1])
            x = round(coordinates[row][2])
            mask_exclude[y - radius:y + radius, x - radius:x + radius] = 0

        df_intensity_background = pd.DataFrame(
            measure.regionprops_table(mask_exclude, images, properties=('label', 'area',),
                                      extra_properties=(mean, sd, median, mad, integrated_intensity,)))

        df_intensity = pd.merge(df_intensity_complete, df_intensity_background, on=['label'],
                                suffixes=('_completecell', '_excludedareacell'))

    else:
        df_intensity = df_intensity_complete.add_suffix('_completecell')
        df_intensity.rename(columns={'label_completecell': 'label'}, inplace=True)

    df_intensity = df_intensity.rename(columns={'label': 'track_id', 'area': 'cellarea'})

    return df_intensity


def mean(regionmask, intensity):
    return np.mean(intensity[regionmask])


def sd(regionmask, intensity):
    return np.std(intensity[regionmask])


def median(regionmask, intensity):
    return np.median(intensity[regionmask])


def mad(regionmask, intensity):
    return stats.median_abs_deviation(intensity[regionmask])


def integrated_intensity(regionmask, intensity):
    return np.sum(intensity[regionmask])
