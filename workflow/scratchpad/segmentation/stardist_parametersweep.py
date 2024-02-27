"""
Parameter sweep for stardist segmentation on 4 example images of a 4D image (tzyx)
"""

# import packages
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from csbdeep.utils import normalize
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from stardist.models import StarDist2D
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


def run_stardist_sweep(imageseries, resizevalue, stardist_prob_threshs, stardist_nms_threshs, path_output):
    """
    Parametersweeo Stardist
    This script takes 3D-image series (tyx) with a pseudo-nuclear staining, selects 4 images and segments the nuclei
    using stardist. Set-up to sweep over the parameters prob_thresh and nms_thresh.

    Args:
         imageseries: (ndarray) Image series from which 4 timepoints are selected for segmentation
         resizevalue: (int) Resizes the image to this given format before segmentation (given in pixel). If 0, no
                      resizing is performed
         stardist_prob_threshs: (list) Probability threshold to be tested. Cellpose uses this value for
                             thresholding shape recognition. The default value for the '2D_versatile_fluo' model is
                             stardist_prob_thresh = 0.48
         stardist_nms_threshs: (list) Overlap threshold to be tested. The default value for the '2D_versatile_fluo'
                                    model is stardist_nms_thresh = 0.3
         path_output: (str) absolute path to where the output should be saved

    Output:
         Overview image: (*.png) original image-label-overlays are saved in an overview image
    """

    # choose 4 positions in the timeseries for testing
    timepoints = [0, 50, 100, 140]
    images = imageseries[timepoints, :, :]

    # Resize images
    if resizevalue != 0:
        images_resized = [resize(images[frame, ...], (resizevalue, resizevalue), anti_aliasing=True) for frame in
                          range(images.shape[0])]
        images_resized = np.array(images_resized)
    else:
        images_resized = images.copy()

    # normalize images
    images_resized = normalize(images_resized)

    # Combine images into one image
    img1 = combine_vertically(images_resized[0, :, :], images_resized[1, :, :])
    img2 = combine_vertically(images_resized[2, :, :], images_resized[3, :, :])
    img_combined = combine_horizontally(img1, img2)

    # Define cellpose model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # actual parameter sweep
    nuclei_masks = []
    for stardist_prob_thresh in stardist_prob_threshs:
        # Gather all images with the current stardist_prob_thresh into a list
        subset = []
        for stardist_nms_thresh in stardist_nms_threshs:
            mask, details = model.predict_instances(img_combined,
                                                    prob_thresh=stardist_prob_thresh,
                                                    nms_thresh=stardist_nms_thresh)
            subset.append(mask)
        # Append list of all images with current stardist_prob_thresh to list of lists
        nuclei_masks.append(subset)

    # plotting
    fig, axes = plt.subplots(len(stardist_prob_threshs), len(stardist_nms_threshs), figsize=(30, 30))

    # Turn off axis
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add row labels
    for ax, thr in zip(axes[:, 0], stardist_prob_threshs):
        ax.set_ylabel("{}".format(np.round(thr, 2)), size=20)

    # Add column labels
    for ax, thr in zip(axes[0], stardist_nms_threshs):
        ax.set_title("{}".format(thr), size=20)

    # Add overlays of images and labels
    for i in range(len(stardist_prob_threshs)):
        for j in range(len(stardist_nms_threshs)):
            # Colour the nuclei
            nuclei_coloured = colour_nuclei(nuclei_masks[i][j])
            # Plot segmentation on top of image
            axes[i, j].imshow(img_combined, cmap="gray", clim=[0, 1.2])
            axes[i, j].imshow(nuclei_coloured, alpha=0.4)

    # Add "title" to signify what columns are
    plt.suptitle("Resize to = {} pixels".format(resizevalue), fontsize=25, y=0.98)

    # Create "outer" image in order to add a common y-label
    fig.add_subplot(111, frameon=False)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Probability threshold", fontsize=25)
    plt.title('Overlap threshold (nms)', fontsize=25)

    # Add bounding box to tight_layout because subtitle is ignored and will therefore overlap with plot otherwise
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    plt.savefig(Path(path_output, "Stardist_maxproj_threshold-matrix_resize-{}.png".format(resizevalue)))
    plt.close('all')


if __name__ == '__main__':
    # give information where to find the image for testing
    absolute_path = '/tungstenfs/scratch/ggiorget/Jana/Microscopy/Mobilisation-E10/Processed/20230614_30s/20230614_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF646_6A12__ntime1_position3_1_FullseqTIRF-Cy5-mCherryGFPWithSMB_s1_MAX.tiff'
    path = os.path.split(absolute_path)[0]
    filename = os.path.split(absolute_path)[1]
    # load image and do a projection
    images = io.imread(absolute_path)
    #images = np.max(images_4d, axis=1)

    # define where to save the output
    abspath_output = os.path.join(path, 'parametersweep_stardist')
    if not os.path.exists(abspath_output):
        # Create a new directory because it does not exist
        os.makedirs(abspath_output)

    # Since the function sweeps over two parameters, run a loop to test a third parameter (resize)
    resize_list = [0, 256, 128, 64]
    for k in tqdm(resize_list):
        run_stardist_sweep(imageseries=images, resizevalue=k,
                           stardist_prob_threshs=[0.28, 0.38, 0.48, 0.58, 0.68],
                           stardist_nms_threshs=[0.1, 0.2, 0.3, 0.4, 0.5], path_output=abspath_output)
