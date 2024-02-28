"""
Max projections
====================
This script takes time course data saved as *.nd and *.stk, loads the time course, performs max projection and saved
resulting images as tiff

author: Jana Tuennermann
"""

# import packages
import argparse
import os
from glob import glob
from os.path import join, basename, dirname

import numpy as np
import pandas as pd
from skimage import io


def main(path_ndimage, path_output):
    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # list all files with given basename and *.stk into a dataframe
    path = path_ndimage.replace('.nd', '')
    files = glob(join(dirname(path), '*' + basename(path) + '*' + '.stk'))
    files = pd.DataFrame(files, columns=['filename'])
    # create columns for basename (excluding time point, referring to single movie) and time point, sort by them
    files[['basename', 'timepoint']] = files['filename'].str.rsplit('_', n=1, expand=True)
    files['timepoint'] = files['timepoint'].str.replace('.stk', '', regex=True).str.replace('t', '', regex=True).astype(
        int)
    files = files.sort_values(['basename', 'timepoint']).reset_index(drop=True)

    # for every movie, load time point, max project and append to 3D array (tyx), save to given output folder
    for movie_name, group in files.groupby(['basename']):
        timpoint_list = group['filename'].tolist()
        images = []
        for timepoint in timpoint_list:
            image = io.imread(timepoint)
            image_max = np.max(image, axis=0)
            images.append(image_max)
        images = np.stack(images, axis=0)
        basename_file = basename(movie_name[0])
        io.imsave(join(path_output, basename_file + '_MAX.tiff'), images, check_contrast=False)


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
