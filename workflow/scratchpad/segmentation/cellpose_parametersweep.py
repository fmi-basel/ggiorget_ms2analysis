"""
Parameter sweep for cellpose segmentation on 4 example images of a 4D image (tzyx)
"""

# import packages
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cellpose import models
from skimage import io
from skimage.exposure import rescale_intensity
from tqdm import tqdm


# helper functions
def rescale_image(img, min_quant=0, max_quant=0.99):
    img = img * 1.0  # turn type to float before rescaling
    min_val = np.quantile(img, min_quant)
    max_val = np.quantile(img, max_quant)
    img = rescale_intensity(img, in_range=(min_val, max_val))
    return img


def combine_vertically(img1, img2, border=15):
    width = img1.shape[1]
    barrier = np.zeros((border, width), np.uint16)
    img = np.append(img1, barrier, axis=0)
    img = np.append(img, img2, axis=0)
    return img


def combine_horizontally(img1, img2, border=15):
    height = img1.shape[0]
    barrier = np.zeros((height, border), np.uint16)
    img = np.append(img1, barrier, axis=1)
    img = np.append(img, img2, axis=1)
    return img


def colour_nuclei(nuclei):
    coloured = np.zeros((nuclei.shape[0], nuclei.shape[1], 3), np.uint8)
    for n in range(nuclei.max()):
        pixels = (nuclei == n + 1)
        coloured[pixels, :] = np.random.randint(1, 255, 3)
    # Add alpha channel to make background transparent
    alpha = np.all(coloured != 0, axis=2) * 255
    rgba = np.dstack((coloured, alpha)).astype(np.uint8)
    return rgba


def run_cellpose_sweep(imageseries, diam, flow_thresh, cellprob_thresh, path_output):
    """
    Parametersweeo Cellpose
    This script takes 3D-image series (tyx) with a pseudo-nuclear staining, selects 4 images and segments the nuclei
    using cellpose. Set-up to sweep over the parameters flow_threshold and cellprob_threshold.

    Args:
         imageseries: (ndarray) Image series from which 4 timepoints are selected for segmentation
         diam: (int) estimate diameter of cells/nuclei in pixel. The original model was trained on cells with a
                         diameter mean of 17 pixel. For automated estimation set diameter = None
         flow_thresh: (list) Flow threshold or model fit threshold to be tested. Cellpose uses this value for
                             thresholding shape recognition. It's default value is 0.4. Should be higher if not enough
                             ROIs are recognised and lower if too many are recognised.
         cellprob_thresh: (list) Cell probability threshold to be tested. The default is cellprob_threshold=0.0.
                                 Decrease this threshold if cellpose is not returning as many ROIs as you’d expect.
                                 Similarly, increase this threshold if cellpose is returning too ROIs particularly from
                                 dim areas. (ranges from -6 to 6)
         path_output: (str) absolute path to where the output should be saved

    Output:
         Overview image: (*.png) original image-label-overlays are saved in an overview image
    """

    # choose 4 positions in the timeseries for testing
    timepoints = [0, 100, 230, 350]
    images = imageseries[timepoints, :, :]

    # Combine images into one image
    img1 = combine_vertically(images[0, :, :], images[1, :, :])
    img2 = combine_vertically(images[2, :, :], images[3, :, :])
    img_combined = combine_horizontally(img1, img2)

    # Define cellpose model
    model = models.Cellpose(gpu=False, model_type='nuclei')

    # actual parameter sweep
    nuclei_masks = []
    for flow_thr in flow_thresh:
        # Gather all images with the current flow_thr into a list
        subset = []
        for cellprob_thr in cellprob_thresh:
            mask, flows, styles, diams = model.eval(img_combined, diameter=diam, channels=[0, 0], do_3D=False,
                                                    flow_threshold=flow_thr, cellprob_threshold=cellprob_thr)
            subset.append(mask)
        # Append list of all images with current flow_thr to list of lists
        nuclei_masks.append(subset)

    # plotting
    fig, axes = plt.subplots(len(flow_thresh), len(cellprob_thresh), figsize=(30, 30))

    # Turn off axis
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add row labels
    for ax, thr in zip(axes[:, 0], flow_thresh):
        ax.set_ylabel("{}".format(np.round(thr, 2)), size=20)

    # Add column labels
    for ax, thr in zip(axes[0], cellprob_thresh):
        ax.set_title("{}".format(thr), size=20)

    # Add overlays of images and labels
    for i in range(len(flow_thresh)):
        for j in range(len(cellprob_thresh)):
            # Colour the nuclei
            nuclei_coloured = colour_nuclei(nuclei_masks[i][j])
            # Plot segmentation on top of image
            axes[i, j].imshow(rescale_image(img_combined), cmap="gray", clim=[0, 1.2])
            axes[i, j].imshow(nuclei_coloured, alpha=0.4)

    # Add "title" to signify what columns are
    plt.suptitle("Diameter = {}".format(diam), fontsize=25, y=0.98)

    # Create "outer" image in order to add a common y-label
    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Flow threshold", fontsize=25)
    plt.title('Cell probability threshold', fontsize=25)

    # Add bounding box to tight_layout because subtitle is ignored and will therefore overlap with plot otherwise
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    plt.savefig(Path(path_output, "CellPose_Model-cyto2_threshold-matrix_diam-{}.png".format(diam)))
    plt.close('all')


if __name__ == '__main__':
    # give information where to find the image for testing
    absolute_path = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20221027_10s/20221027_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF646i_6A12_1_FullseqTIRF-Cy5-mCherryGFPWithSMB_s1.tif'
    path = os.path.split(absolute_path)[0]
    filename = os.path.split(absolute_path)[1]
    # load image and do a projection
    images_4d = io.imread(absolute_path)
    images_median = np.median(images_4d, axis=1)
    # define where to save the output
    abspath_output = os.path.join(path, 'parametersweep')
    if not os.path.exists(abspath_output):
        # Create a new directory if it does not exist
        os.makedirs(abspath_output)

    # Since the function sweeps over two parameters, run a loop to test a third parameter (diameter)
    diameters = [80, 90, 100, 120]
    for diameter in tqdm(diameters, ascii=True):
        run_cellpose_sweep(imageseries=images_median, diam=diameter, flow_thresh=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
                           cellprob_thresh=[-1., -2., -3., -4., -5., -6.], path_output=abspath_output)
