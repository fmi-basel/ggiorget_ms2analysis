"""
Intensity read-out for MS2 Image Analysis Pipeline (mask)
====================
This script takes 2D-time course (xyt) of MS2 labeled images and corresponding tracks and reads out the spot intensity
over time using a circular mask. It also reads out the intensity of the whole cell and GFP intensity.
The general workflow includes:
1. data loading, pre-processing (flat-field correction)
2. read-out of spot intensities
3. read-out whole cell intensity
4. calculate bleach corrected intensities
5. read-out GFP intensity

author: Jana Tuennermann
"""

# Let's start with loading all the packages and modules you need for this script
import argparse
import os
import sys

import numpy as np
import pandas as pd
from skimage import io

from global_cell_readout import wholecell_readout_timeseries, wholecell_readout_singleframe


def flatfieldcorrection(image, flatfieldimage, darkimage):
    """
    flat-field correction using following approach: corrected image = image * (mean flat-field image) /flat-field image
    Args:
         image: (ndarray) image to be corrected
         flatfieldimage: (ndarray) flat-field image for correction
         darkimage: (ndarray) dark-image for correction (detector noise)

    Returns:
         image: (ndarray) corrected image
    """
    image = (image - darkimage) * np.mean(flatfieldimage - darkimage) / (flatfieldimage - darkimage)
    return image


def GaussianMaskFit2(im, coo, s, optLoc=1, bgSub=2, winSize=13, convDelta=.01, nbIter=20):
    """Applies the algorithm from [Thompson et al. (2002) PNAS, 82:2775].
    Parameters:
    - im: a numpy array with the image
    - coo: approximate coordinates (in pixels) of the spot to localize and measure. Note, the coordinates are x,y!
    - s: width of the PSF in pixels
    - optLoc: If 1, applied the iterative localization refinement algorithm, starting with the coordinates provided in coo. If 0, only measures the spot intensity at the coordinates provided in coo.
    - bgSub: 0 -> no background subtraction. 1 -> constant background subtraction. 2 -> tilted plane background subtraction.
    - winSize: Size of the window (in pixels) around the position in coo, used for the iterative localization and for the background subtraction.
    - convDelta: cutoff to determine convergence, i.e. the distance (in pixels) between two iterations
    - nbIter: the maximal number of iterations.

    Returns
    - the intensity value of the spot.
    - the coordinates of the spot.

    If convergence is not found after nbIter iterations, return 0 for both intensity value and coordinates.
    """
    coo = np.asarray(coo)
    for i in range(nbIter):
        if not np.prod(coo - winSize / 2 >= 0) * np.prod(coo + winSize / 2 <= im.shape[::-1]):
            break
        winOrig = (coo - int(winSize) // 2).astype(int)
        i, j = np.meshgrid(winOrig[0] + np.r_[:winSize], winOrig[1] + np.r_[:winSize])
        N = np.exp(-(i - coo[0]) ** 2 / (2 * s ** 2) - (j - coo[1]) ** 2 / (2 * s ** 2)) / (2 * np.pi * s ** 2)
        S = im[:, winOrig[0]:winOrig[0] + winSize][winOrig[1]:winOrig[1] + winSize].astype(float)
        if bgSub == 0:
            bg = 0, 0, 0
        elif bgSub == 1:
            bg = np.mean([S[0], S[-1], S[:, 0], S[:, -1]]), 0, 0
            S -= bg[0]
        else:
            xy = np.r_[:2 * winSize] % winSize - (winSize - 1) / 2
            bgx = np.polyfit(xy, np.r_[S[0], S[-1]], 1)
            S = (S - xy[:winSize] * bgx[0]).T
            bgy = np.polyfit(xy, np.r_[S[0], S[-1]], 1)
            S = (S - xy[:winSize] * bgy[0]).T
            bg = np.mean([S[0], S[-1], S[:, 0], S[:, -1]])
            S -= bg
            bg = bg, bgx[0], bgy[0]

        S = S.clip(0)  # Prevent negative values !!!!

        if optLoc:
            SN = S * N
            ncoo = np.r_[np.sum(i * SN), np.sum(j * SN)] / np.sum(SN)
            # ncoo=ncoo+ncoo-coo # Extrapolation
            if abs(coo - ncoo).max() < convDelta:
                return np.sum(SN) / np.sum(N ** 2), coo, bg
            else:
                coo = ncoo
        else:
            return np.sum(S * N) / np.sum(N ** 2), coo, bg
    return 0, coo, (0,0,0)


def bleach_correction(trace_intensity, cell_intensity):
    """
    simple bleach correction using normalized cell intensity as a bleach factor
    Args:
         trace_intensity: (pd.Series) intensity trace to be corrected
         cell_intensity: (pd.Series) intensity trace of whole cell used for bleach correction

    Returns:
         corr_trace_intensity: (pd.Series) corrected intensity trace
    """
    norm_bleach_factor = cell_intensity / np.max(cell_intensity)
    corr_trace_intensity = trace_intensity / norm_bleach_factor
    return corr_trace_intensity


def main(image_path, tracks_path, mask_image_path, gfp_image_path, flatfield_path, path_output, optimize_gaussian):
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(image_path)[1]
    # 1. ------- Data loading and Pre-processing--------
    images_maxproj = io.imread(image_path)
    df_tracks = pd.read_csv(tracks_path)
    mask_image = io.imread(mask_image_path)
    gfp_image = io.imread(gfp_image_path)

    # Load images for flat-field correction. I use an automated approach to recognize the needed images, so I also
    # check if it works before continuing
    flatfield_images_list = os.listdir(flatfield_path)
    flatfield_cy5_filename = [s for s in flatfield_images_list if "568" in s and "mCherry-GFPCy5" in s]
    darkimage_cy5_filename = [s for s in flatfield_images_list if "Darkimage" in s and "mCherry-GFPCy5" in s]
    flatfield_gfp_filename = [s for s in flatfield_images_list if "488" in s and "GFP-Cy5mCherry" in s]
    darkimage_gfp_filename = [s for s in flatfield_images_list if "Darkimage" in s and "GFP-Cy5mCherry" in s]
    for image in [flatfield_cy5_filename, darkimage_cy5_filename, flatfield_gfp_filename, darkimage_gfp_filename]:
        if len(image) != 1:
            print('Automated selection for flat-flied images did not work. Either naming was incorrect or an '
                  'image is missing')
            sys.exit()
    flatfield_cy5_filename = flatfield_cy5_filename[0]
    darkimage_cy5_filename = darkimage_cy5_filename[0]
    flatfield_gfp_filename = flatfield_gfp_filename[0]
    darkimage_gfp_filename = darkimage_gfp_filename[0]

    flatfield_cy5_image = io.imread(os.path.join(flatfield_path, flatfield_cy5_filename))
    dark_cy5_image = io.imread(os.path.join(flatfield_path, darkimage_cy5_filename))
    flatfield_gfp_image = io.imread(os.path.join(flatfield_path, flatfield_gfp_filename))
    dark_gfp_image = io.imread(os.path.join(flatfield_path, darkimage_gfp_filename))

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # Flat-field correction
    images_corr = np.stack(
        [flatfieldcorrection(images_maxproj[i, ...], flatfield_cy5_image, dark_cy5_image) for i in
         range(images_maxproj.shape[0])],
        axis=0)

    images_gfp_corr = flatfieldcorrection(gfp_image, flatfield_gfp_image, dark_gfp_image)

    # 2. ------- MS2 intensity read-out (from flat-field corrected image) --------
    # The intensity is read/out using a circular mask around the position. The same estimated spot size is used as for
    # spot detection. Additionally, I read out the background intensity, using a ring around the circular mask. It's
    # seperated by a no of pixels (gap size), which I determined before.

    # Here the geometrics I will use
    df_intensity = []
    for index_original, row in df_tracks.iterrows():
        # for indes, row in df_tracks.iloc[1860:1861,:].iterrows():
        intensity, coordinates, tilt = GaussianMaskFit2(im=images_corr[int(row['frame']), :, :], coo=(row['x'], row['y']),
                                                        s=2, optLoc=optimize_gaussian, bgSub=2, winSize=15, convDelta=.01, nbIter=5)
        fit = 'success' if optimize_gaussian else 'intensity'
        if intensity == 0:
            intensity, coordinates, tilt = GaussianMaskFit2(im=images_corr[int(row['frame']), :, :],
                                                            coo=(row['x'], row['y']), s=2, optLoc=False, bgSub=2,
                                                            winSize=15, convDelta=.01, nbIter=5)
            fit = 'fail'
        row_intensity = pd.DataFrame(
            {'intensity': intensity, 'frame': row['frame'], 'x': coordinates[0], 'y': coordinates[1], 'bg_local': tilt[0],
             'bgx_local': tilt[1], 'bgy_local': tilt[2], 'optimized': fit}, index=[index_original])
        df_intensity.append(row_intensity)
    df_intensity = pd.concat(df_intensity)
    df_tracks = df_tracks.drop(['frame', 'y', 'x'], axis=1).merge(df_intensity, left_index=True, right_index=True)


    # 3. ------- Whole cell read-out --------
    df_intensity_completecell = wholecell_readout_timeseries(images_corr, mask_image)
    df_tracks = df_tracks.merge(df_intensity_completecell, how='left')

    # 4. ------- Bleach correction --------
    # Since I already read-out intensity based on background subtracted images, I use the mad of intensity of the whole
    # cell to calculate a bleach factor and correct the intensity traces using this factor
    df_tracks['intensity_corr'] = df_tracks.groupby('track_id').apply(
            lambda cell: bleach_correction(cell.intensity, cell.mad_completecell), include_groups=False).reset_index(
            level=0, drop=True)

    # 5. ------- GFP read-out --------
    # Using the first mask frame, read out the GFP intensity of the whole cell
    print('Reading out GFP')
    df_intensity_gfp = wholecell_readout_singleframe(images_gfp_corr, mask_image[0, :, :])
    df_intensity_gfp.columns = df_intensity_gfp.columns.str.replace('_completecell', '_gfp')
    df_intensity_gfp.drop(columns='area_gfp', inplace=True)
    df_tracks = df_tracks.merge(df_intensity_gfp, how='left')

    # 5. ------- Saving data --------
    df_tracks.to_csv(os.path.join(path_output, images_filename.replace('_MAX.tiff', '_tracks_intensity.csv')),
                     index=False)
    print('Done :)')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Input movie to be used, requires absolute path",
    )
    parser.add_argument(
        "-it",
        "--input_tracks",
        type=str,
        default=None,
        required=True,
        help="path to tracks dataframe corresponding to movie, requires absolute path",
    )
    parser.add_argument(
        "-is",
        "--input_segmentation_image",
        type=str,
        default=None,
        required=True,
        help="Segmentation images to be used, requires absolute path",
    )
    parser.add_argument(
        "-ig",
        "--input_gfpimage",
        type=str,
        default=None,
        required=True,
        help="path to gfp image corresponding to movie, requires absolute path",
    )
    parser.add_argument(
        "-if",
        "--input_flatfield",
        type=str,
        default=None,
        required=True,
        help="Directory in which images for flat-field correction are saved, requires absolute path",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        default=None,
        required=True,
        help="Path to output directory, requires absolute path",
    )
    parser.add_argument(
        "-opt",
        "--opt_gaussian",
        action='store_true',
        help="Gaussian fitting/refinement",
    )
    args = parser.parse_args()

    main(image_path=args.input_path, tracks_path=args.input_tracks, mask_image_path=args.input_segmentation_image,
         gfp_image_path=args.input_gfpimage, flatfield_path=args.input_flatfield, path_output=args.path_output,
         optimize_gaussian=args.opt_gaussian)
