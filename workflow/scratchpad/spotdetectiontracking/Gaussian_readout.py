"""
Bullshit script to try and fit a Gaussian to the spot and read-out intensity this way
"""

import os

import numpy as np
import pandas as pd
from skimage import io
import math

from utils import flatfieldcorrection

# Tell me where the data is and where to save it
absolute_path = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20221124_10s/20221124_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF646_6A12_4_FullseqTIRF-Cy5-mCherryGFPWithSMB_s2_MAX.tif'
path = os.path.split(absolute_path)[0]
path_output = os.path.join(path, 'output')

path_flatfield = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20221125/'
flatfield_cy5_filename = 'AVG_20221125_Alexa647-0.5uM_FullseqTIRF-Cy5-mCherryGFPWithSMB.tif'
darkimage_cy5_filename = 'AVG_20221125_Darkimage_Gain100_1_FullseqTIRF-Cy5-mCherryGFPWithSMB.tif'

# 1. ------- Data loading and a bit of clean-up --------
images_filename = os.path.split(absolute_path)[1]

images_maxproj = io.imread(absolute_path)
flatfield_cy5_image = io.imread(os.path.join(path_flatfield, flatfield_cy5_filename))
dark_cy5_image = io.imread(os.path.join(path_flatfield, darkimage_cy5_filename))

df_tracks = pd.read_csv(os.path.join(path_output, images_filename.replace('_MAX.tif', '_tracks.csv')))

# Flat-field correction
images_corr = np.stack(
    [flatfieldcorrection(images_maxproj[i, ...], flatfield_cy5_image, dark_cy5_image) for i in
     range(images_maxproj.shape[0])],
    axis=0)

df_backup = df_tracks.copy()
image = images_corr[0,...]


def gauss_2d(xy:tuple, amplitude, x0, y0, sigma_xy, offset):
    """2D gaussian."""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
        )
    )
    return gauss

def gauss_1d(x, amplitude, x0, sigma, offset):
    """2D gaussian."""
    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma ** (2)))
        )
    )
    return gauss

x_coord = df_tracks.loc[0, 'x']
y_coord = df_tracks.loc[0, 'y']
def gauss_single_spot(image: np.ndarray, x_coord: float, y_coord: float, background:float, crop_size=17, EPS=1e-4) -> tuple:
    """Gaussian prediction on a single crop centred on spot."""
    import scipy.optimize as opt

    # crop image around the guessed spot
    start_dim1 = np.max([int(np.round(y_coord - crop_size // 2)), 0])
    if start_dim1 < len(image) - crop_size:
        end_dim1 = start_dim1 + crop_size
    else:
        start_dim1 = len(image) - crop_size
        end_dim1 = len(image)

    start_dim2 = np.max([int(np.round(x_coord - crop_size // 2)), 0])
    if start_dim2 < len(image) - crop_size:
        end_dim2 = start_dim2 + crop_size
    else:
        start_dim2 = len(image) - crop_size
        end_dim2 = len(image)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2]

    x = np.arange(0, crop.shape[1], 1)
    y = np.arange(0, crop.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop
    sigma_guess = max(*crop.shape) * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(np.max(crop) / 2, np.min(crop))  # Height of gaussian, maximum value
    offset = np.min(crop)
    initial_guess = [amplitude_max, x0, y0, sigma_guess, offset]

    # Parameter search space bounds
    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, pcov = opt.curve_fit(
            gauss_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
            method='trf'
        )
    except RuntimeError:
        # print('Runtime')
        return sigma

    sigma = popt[3]
    x =popt[1]
    y= popt[2]

def gaussian_kernel_readout(images, coordinates, window_size=7):
    # Define the size and subpixel offset of the kernel


        # Create a meshgrid of x and y values centered at the peak offset and the size of the whole image
        x_offset = np.arange(crop.shape[1]) - x
        y_offset = np.arange(crop.shape[0]) - y
        xx, yy = np.meshgrid(x_offset, y_offset)

        # Compute the Gaussian kernel
        kernel_gauss = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel_gauss /= kernel_gauss.sum()
        kernel_gauss.sum()

        # Apply the Gaussian kernel and read-out intensity
        integreated_intensity = (crop * kernel_gauss).sum()/(kernel_gauss ** 2).sum()


        df_intensity.append([t, y, x, integreated_intensity])



#plot the fit results
array_x = crop[8,...]
array_y = crop[...,8]
gauss_x = [*popt]
del gauss_x[2]
gauss_y = [*popt]
del gauss_y[1]


plt.plot(x,gauss_1d(x, *gauss_x), label='Gaussian fit')
plt.plot(x,array_x, label='raw data')
plt.legend()
plt.ylabel('Intensity (a.u.)')
plt.xlabel('position (px)')
plt.show()

area = popt[0] * (popt[3] * math.sqrt(2 * math.pi))
