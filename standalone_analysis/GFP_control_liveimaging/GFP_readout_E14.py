"""
For the GFP read-out of my MS2 imaging, I wanted to have a negative control. I used the E14ddCTCF cell line, which is
GFP negative, labeled their nuclei using SirDNA and imaged them using the same settings as for the MS2 imaging. In this
script, I analyse the data the same way as for the MS2 imaging:
1. Load images
2. Mean projection for GFP, max projection for Cy5 (nuclei)
3. Segmentation of nuclei using stardist and Cy5 channel
4. Flat-field correction for GFP images
5. Read-out of GFP intensity
6. Add some useful information about the experiment
7. Save the data as csv file and the label images as tif files
"""

# Load all the packages
import os

import numpy as np
from csbdeep.utils import normalize
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.transform import resize
from stardist.models import StarDist2D

from workflow.scripts.s02_detectionandtracking.global_cell_readout import wholecell_readout_timeseries
from workflow.scripts.utils import flatfieldcorrection

# 1. ------- Load images --------
path = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Raw/20230724_E14_GFP'
# list all files in the folder that ends with .stk in their name and correspond either to GFP or Cy5
filenames_GFP = [os.path.join(path, filename) for filename in os.listdir(path) if
                 filename.endswith('.stk') and 'w1' in filename]
filenames_Cy5 = [os.path.join(path, filename) for filename in os.listdir(path) if
                 filename.endswith('.stk') and 'w2' in filename]
# sort them alphabetically, so that the order is the same for GFP and Cy5
filenames_GFP.sort()
filenames_Cy5.sort()

# load all files
images_GFP = []
for filename in filenames_GFP:
    image = io.imread(filename)
    images_GFP.append(image)

images_Cy5 = []
for filename in filenames_Cy5:
    image = io.imread(filename)
    images_Cy5.append(image)

# 2. ------- Projection --------
images_GFP = np.asarray([np.mean(images_GFP[i], axis=0) for i in range(len(images_GFP))])
images_Cy5 = np.asarray([np.max(images_Cy5[i], axis=0) for i in range(len(images_Cy5))])

# 3. ------- Segmentation based on Cy5 channel using stardist --------
# creates a pretrained model for stardist, I use the '2D_versatile_fluo' model as default
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# Resize image
resizevalue = 128
images_Cy5_resized = np.asarray(
    [resize(images_Cy5[frame, ...], (resizevalue, resizevalue), anti_aliasing=True) for frame in
     range(images_Cy5.shape[0])])

# actual segmentation using default values
labels_resized, details = zip(*[
    model.predict_instances(normalize(images_Cy5_resized[frame, ...])) for frame in range(images_Cy5_resized.shape[0])])
labels_resized = np.asarray(labels_resized)

# resize back
labels = np.asarray(
    [resize(labels_resized[frame, ...], (images_Cy5.shape[1], images_Cy5.shape[1]), order=0)
     for frame in range(labels_resized.shape[0])])

# filter small objects and remove cells touching the border
labels = np.asarray([remove_small_objects(labels[frame, ...], min_size=2000) for frame in range(labels.shape[0])])
labels = np.asarray([clear_border(labels[frame, ...]) for frame in range(labels.shape[0])])

# 4. ------- Flat-field correction for GFP images --------
# Load images for flat-field correction
path_flatfield = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/Flatfield/20230724/'

flatfield_images_list = os.listdir(path_flatfield)
flatfield_gfp_filename = [s for s in flatfield_images_list if "488" in s and "GFP-Cy5mCherry" in s]
darkimage_gfp_filename = [s for s in flatfield_images_list if "Darkimage" in s and "GFP-Cy5mCherry" in s]
flatfield_gfp_image = io.imread(os.path.join(path_flatfield, flatfield_gfp_filename[0]))
dark_gfp_image = io.imread(os.path.join(path_flatfield, darkimage_gfp_filename[0]))

images_GFP_corr = np.stack(
    [flatfieldcorrection(images_GFP[i, ...], flatfield_gfp_image, dark_gfp_image) for i in
     range(images_GFP.shape[0])],
    axis=0)

# 5. ------- GFP read-out --------
# read out the GFP intensity of the whole cell
df_intensity_gfp = wholecell_readout_timeseries(images_GFP_corr, labels)
df_intensity_gfp = df_intensity_gfp.rename(columns={'mean': 'mean_GFP', 'sd': 'sd_GFP', 'median': 'median_GFP',
                                                    'mad': 'mad_GFP', 'sumup': 'integratedINT_GFP'})

# 6. ------- Add some useful information about the experiment --------
# add filename: frame corresponds to position of filename in filenames_GFP
df_intensity_gfp['filename_gfp'] = [os.path.split(filenames_GFP[position])[1] for position in df_intensity_gfp['frame']]
df_intensity_gfp['filename'] = [os.path.split(filenames_Cy5[position])[1] for position in df_intensity_gfp['frame']]
df_intensity_gfp['filename flat-field GFP'] = flatfield_gfp_filename[0]
# set frame to 0 (timepoint 0 now and not position in filenames_GFP)
df_intensity_gfp['frame'] = 0
# unique id and clone name
df_intensity_gfp['unique_id'] = df_intensity_gfp.groupby(['filename', 'track_id']).ngroup()
df_intensity_gfp['clone'] = 'E14ddCTCF'

# 7. ------- Save data --------
path_out = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/'
io.imsave(os.path.join(path_out, '20230724_E14_GFP', '20230724_E14ddCTCF_label-images.tif'), labels,
          check_contrast=False)
df_intensity_gfp.to_csv(os.path.join(path_out, 'output/fiveclones', '20230724_E14ddCTCF_GFP.csv'), index=False)
