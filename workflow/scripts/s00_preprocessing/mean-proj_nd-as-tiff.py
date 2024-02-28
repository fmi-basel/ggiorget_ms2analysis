"""
Max projections
====================
This script takes multi-stage z-stack data saved as *.nd and *.stk, loads the files, performs mean projection and saves
resulting images as tiff.

author: Jana Tuennermann
"""

# import packages
import argparse
from glob import glob
from os.path import join, basename, dirname

import numpy as np
from skimage import io


def main(path_ndimage, path_output):
    # list all files with given basename and *.stk into a dataframe
    path = path_ndimage.replace('.nd', '')
    files = glob(join(dirname(path), '*' + basename(path) + '*' + '.stk'))

    # for every movie, load time point, max project and append to 3D array (tyx), save to given output folder
    for file in files:
        image = io.imread(file)
        image_mean = np.mean(image, axis=0)
        filename = basename(file).replace('.stk', '_MEAN.tiff')
        io.imsave(join(path_output, filename), image_mean, check_contrast=False)


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_ndimage",
        type=str,
        default=None,
        required=True,
        help="nd image to be processed, requires absolute path",
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

    main(path_ndimage=args.input_ndimage, path_output=args.path_output)
