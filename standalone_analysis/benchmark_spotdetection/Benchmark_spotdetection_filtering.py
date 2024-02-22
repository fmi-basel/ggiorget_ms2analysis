"""
During spot detection, I can filter for some spot features. Here, for ground truth data I have, I try to find robust
thresholds for these spot features.
Workflow:
1. load data, preprocess as for the main workflow (local background sub, spot detection, assignment to cells)
2. plot spot features and encode if the spot is a real one
"""
# Let's start with loading all the packages
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from skimage import io

from utils import local_backgroundsubtraction

tp.quiet()

path_output = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/plotting_figures_movies/spotdetection_filter'

# -- load data + preprocess --
path_images = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/processed/20230715_30s'
filename = '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_MAX.tiff'
images = io.imread(os.path.join(path_images, 'proj_ms2', filename))
images = images[0:601, :, :]

mask_labelimagename = filename.replace('_MAX.tiff', '_label-image.tif')
mask_image = io.imread(os.path.join(path_images, 'segmentation', mask_labelimagename))
mask_csvname = filename.replace('_MAX.tiff', '_label-image_tracks.csv')
mask_track = pd.read_csv(os.path.join(path_images, 'segmentation', mask_csvname))

path_groundtruth = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/data/groundtruth_tracks'
filename_groundtruth = filename.replace('_MAX.tiff', '_cuttracks_groundtruth.csv')
df_groundtruth = pd.read_csv(os.path.join(path_groundtruth, filename_groundtruth))
df_groundtruth = df_groundtruth.round(0)

# local background subtraction of images, works better for spot detection later
images_sub = np.stack(
    [local_backgroundsubtraction(images[i, ...], pixelsize=5) for i in range(images.shape[0])],
    axis=0)

# spot detection
df_spots = tp.batch(images_sub, diameter=9, minmass=4600)

# assign if detected spots are true
df_spots = df_spots.round({'y': 0, 'x': 0})
df_spots = df_spots.merge(round(df_groundtruth[['frame', 'x', 'y']]), on=['frame', 'x', 'y'],
                          how='left', indicator=True)
df_spots = df_spots.rename(columns={"_merge": "groundtruth"})
df_spots = df_spots.replace({'both': True, 'right_only': True, 'left_only': False})

# Assign spots to cells (Label image ID)
t = df_spots['frame'].astype(np.int64)
y = df_spots['y'].astype(np.int64)
x = df_spots['x'].astype(np.int64)
df_spots['track_id'] = mask_image[t, y, x]

# Let's get rid of spots which are not assigned to cells
df_spots = df_spots.merge(mask_track, on=['track_id', 'frame'])

# -- Plotting --
colors = {True: 'black', False: 'red'}
columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'cellarea', 'mean', 'sd', 'median', 'mad', 'sumup']
unique_combinations = set(combinations(columns, 2))

for feature1, feature2 in unique_combinations:
    for value, color in colors.items():
        mask = df_spots['groundtruth'] == value
        plt.scatter(df_spots[mask][feature1], df_spots[mask][feature2], c=color, label=str(value))
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.legend()
    # plt.savefig(os.path.join(path_output, f'detection_{feature1}_{feature2}.pdf'))
    plt.show()
    plt.close()

# Threshold on mass, size
threshold_size = 1.29  # default 1.19 for JF646, 1.29 for JF549
threshold_mass = 30000  # default 21000 for JF646, 30000 for JF549

df_spots_filtered = df_spots[df_spots['mass'] <= threshold_mass]
df_spots_filtered = df_spots_filtered[df_spots_filtered['size'] >= threshold_size]

for feature1, feature2 in unique_combinations:
    for value, color in colors.items():
        mask = df_spots_filtered['groundtruth'] == value
        plt.scatter(df_spots_filtered[mask][feature1], df_spots_filtered[mask][feature2], c=color, label=str(value))
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.legend()
    # plt.savefig(os.path.join(path_output, 'After_thresholdsizemass', f'detection_{feature1}_{feature2}.pdf'))
    plt.show()
    plt.close()


# PCA on spot features
from sklearn.decomposition import PCA

featues_spotdetection = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']

num_eigvectors = 2

# Standardize data
df_spots_norm = df_spots_filtered[featues_spotdetection]
df_spots_norm = df_spots_norm - df_spots_norm.mean()
df_spots_norm = df_spots_norm / df_spots_norm.std()

# PCA
pca = PCA(n_components=num_eigvectors)
pca.fit(df_spots_norm)
pca_components = pca.transform(df_spots_norm)
pca_components = pd.DataFrame(pca_components, columns=[f'PC{i}' for i in range(1, num_eigvectors + 1)])
pca_components['groundtruth'] = df_spots['groundtruth']

print(pca.explained_variance_ratio_)

# Plot
for value, color in colors.items():
    mask = pca_components['groundtruth'] == value
    plt.scatter(pca_components[mask]['PC1'], pca_components[mask]['PC6'], c=color, label=str(value))
plt.xlabel(f'PC1')
plt.ylabel(f'PC2')
plt.legend()
#plt.savefig(os.path.join(path_output, 'PCA_filtered.pdf'))
plt.show()
plt.close()

# Plot
for value, color in colors.items():
    mask = pca_components['groundtruth'] == value
    plt.scatter(pca_components[mask]['PC1'], pca_components[mask]['PC3'], c=color, label=str(value))
plt.xlabel(f'PC1')
plt.ylabel(f'PC3')
plt.legend()
plt.savefig(os.path.join(path_output, 'PCA.pdf'))
plt.show()
plt.close()
