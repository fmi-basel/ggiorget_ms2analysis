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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
from skimage import io

def local_backgroundsubtraction(image, pixelsize):
    """
    local background subtraction using a round kernel and a defined pixel size
    Args:
         image: (ndarray) image to be background subtracted
         pixelsize: (int) kernel size in pixels, need to be an uneven number

    Returns:
         image: (ndarray) background subtracted image
    """
    from skimage.morphology import white_tophat
    from skimage.morphology import disk
    image = white_tophat(image, disk(pixelsize))
    return image


def linear_sum_colocalisation(spot_coord_1: pd.DataFrame, spot_coord_2: pd.DataFrame, cutoff_dist: int):
    """
    Compare two dataframes with spot coordinates and indicate if spot is present in both dataframes with a certain
    cutoff distance.

    Parameters
    ----------
    spot_coord_1 :
        Dataframe with first set of spot coordinates

    spot_coord_2 :
        Dataframe with second set of spot coordinates

    cutoff_dist :
        cutoff distance in pixels (sum over all axis)

    Returns
    -------
    input dataframes with additional column indicating co-localisation and one additional dataframe with matched spots
    """
    spot_coord_1_indexed = spot_coord_1.reset_index()
    spot_coord_1_indexed.rename(columns={'index': 'index_spot'}, inplace=True)
    spot_coord_2_indexed = spot_coord_2.reset_index()
    spot_coord_2_indexed.rename(columns={'index': 'index_groundtruth'}, inplace=True)

    spot_coord_1_values = spot_coord_1_indexed[['y', 'x']].dropna().values
    spot_coord_2_values = spot_coord_2_indexed[['y', 'x']].dropna().values

    # if one of the dataframes is empty, cdist can cause an error, if that's the case, return empty indexes
    try:
        # calculate the distance matrix and find the optimal assignment
        global_distances = cdist(spot_coord_1_values, spot_coord_2_values, 'euclidean')
        spot_coord_1_ind, spot_coord_2_ind = linear_sum_assignment(global_distances)

        # use threshold on distance
        assigned_distance = spot_coord_1_values[spot_coord_1_ind] - spot_coord_2_values[spot_coord_2_ind]
        distance_id = np.sum(np.abs(assigned_distance), axis=1)

        spot_coord_1_ind = spot_coord_1_ind[distance_id < cutoff_dist]
        spot_coord_2_ind = spot_coord_2_ind[distance_id < cutoff_dist]
    except ValueError:
        spot_coord_1_ind = []
        spot_coord_2_ind = []

    # create a dataframe only containing the matches spots, where one row corresponds to a match
    match_index_df = pd.DataFrame({'match_spot': spot_coord_1_ind, 'match_groundtruth': spot_coord_2_ind})

    match_df = pd.merge(match_index_df, spot_coord_1_indexed, left_on='match_spot', right_index=True, how='outer')
    match_df = pd.merge(match_df, spot_coord_2_indexed, left_on='match_groundtruth', right_index=True, how='outer')
    match_df = match_df[['index_spot', 'index_groundtruth']].reset_index(drop=True)

    return match_df

def signallength(x, column):
    """
    From a series calculate how long a state was present.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
    Returns:
         burst_stats: (pd.Dataframe) processed dataframe
    """
    x = x.assign(signal_no=(x[column] != x[column].shift()).cumsum())
    burst_stats = x.groupby(['unique_id', 'signal_no', column]).agg(
        {'frame': ['count']}).reset_index()
    burst_stats.columns = list(map(''.join, burst_stats.columns.values))
    return burst_stats


tp.quiet()

path_output = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/benchmarking/spotdetection_filter'

# -- load data + preprocess --
path_groundtruth = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/groundtruth_halfenh/'
filename_groundtruth = [
    '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
    '20230719_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5E10_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
'20230720_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5F11_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s5_groundtruth.csv'
]
"""
filename_groundtruth = [
    '20240131_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_4A1_2_FullseqTIRF-mCherry-GFPCy5WithSMB_s3_groundtruth.csv',
    '20240201_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_4A1_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s4_groundtruth.csv',
    '20240201_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
    '20240203_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_2_FullseqTIRF-mCherry-GFPCy5WithSMB_s2_groundtruth.csv'
]
"""
# Load groundtruth data
df_groundtruth = []
for filename in filename_groundtruth:
    data = pd.read_csv(os.path.join(path_groundtruth,filename))
    data['filename'] = filename.replace('_groundtruth.csv', '_MAX.tiff')
    df_groundtruth.append(data)
df_groundtruth = pd.concat(df_groundtruth)
#df_groundtruth = df_groundtruth.rename(columns={'axis-0': 'frame', 'axis-1': 'y', 'axis-2': 'x'})

# Name corresponding images
path_images = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/processed/'
filename_images = [filename.replace('_groundtruth.csv', '_MAX.tiff') for filename in filename_groundtruth]


# perform spot detection
df_detection = []
for file in filename_images:
    date = file.split('_')[0]
    images = io.imread(os.path.join(path_images, date, 'proj', file))
    images = images[0:601, :, :]

    mask_labelimagename = file.replace('_MAX.tiff', '_label-image.tiff')
    mask_image = io.imread(os.path.join(path_images, date, 'segmentation', mask_labelimagename))


    # local background subtraction of images, works better for spot detection later
    images_sub = np.stack(
        [local_backgroundsubtraction(images[i, ...], pixelsize=5) for i in range(images.shape[0])],
        axis=0)

    # spot detection
    df_spots = tp.batch(images_sub, diameter=7, minmass=3000)


    # Assign spots to cells (Label image ID)
    t = df_spots['frame'].astype(np.int64)
    y = df_spots['y'].astype(np.int64)
    x = df_spots['x'].astype(np.int64)
    df_spots['track_id'] = mask_image[t, y, x]

    # Let's get rid of spots which are not assigned to cells
    df_spots = df_spots[df_spots.track_id != 0]
    df_spots['filename'] = file
    df_detection.append(df_spots)
df_detection = pd.concat(df_detection)

# compare to ground truth
df_comparision = []
for file in filename_images:
    df_spots = df_detection[df_detection['filename'] == file]
    df_groundtruth_file = df_groundtruth[df_groundtruth['filename'] == file]

    unique_frames = np.unique(pd.concat([df_spots['frame'], df_groundtruth_file['frame']]))
    df_1 = []
    for frame in unique_frames:
        spots_frame = df_spots[df_spots['frame'] == frame]
        groundtruth_frame = df_groundtruth_file[df_groundtruth_file['frame'] == frame]

        df_2 = linear_sum_colocalisation(spots_frame, groundtruth_frame, 2)
        # add frame information
        df_2['frame'] = frame
        df_1.append(df_2)
    df_1 = pd.concat(df_1)
    df_1['filename'] = file
    df_comparision.append(df_1)
df_comparision = pd.concat(df_comparision)

df_comparision = df_comparision[df_comparision['index_spot'].notna()]
df_comparision['correct'] = df_comparision.iloc[:, [0,1]].notnull().all(axis=1)
df_merge = df_detection.reset_index()
df_merge.rename(columns={'index': 'index_spot'}, inplace=True)
df_comparision = df_comparision.merge(df_merge, on=['filename', 'index_spot', 'frame'])

# -- Plotting --
colors = {True: 'black', False: 'red'}
columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame']
unique_combinations = set(combinations(columns, 2))

for feature1, feature2 in unique_combinations:
    for value, color in colors.items():
        mask = df_comparision['correct'] == value
        plt.scatter(df_comparision[mask][feature1], df_comparision[mask][feature2], c=color, label=str(value))
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.legend()
    #plt.savefig(os.path.join(path_output, f'detection_{feature1}_{feature2}.pdf'))
    plt.show()
    plt.close()

# Threshold on mass, size
threshold_size_min = 1.2
threshold_size_max = 1.7 # default 1.19 for JF646, 1.29 for JF549
threshold_mass = 30000  # default 21000 for JF646, 30000 for JF549

df_comparision_filtered = df_comparision[(df_comparision['mass'] <= threshold_mass) & (df_comparision['size'] >= threshold_size_min) & (df_comparision['size'] <= threshold_size_max)]
#df_comparision_filtered = df_comparision_filtered[df_comparision_filtered['size'] >= threshold_size_min]

for feature1, feature2 in unique_combinations:
    for value, color in colors.items():
        mask = df_comparision_filtered['correct'] == value
        plt.scatter(df_comparision_filtered[mask][feature1], df_comparision_filtered[mask][feature2], c=color, label=str(value))
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.legend()
    plt.savefig(os.path.join(path_output, f'After_thresholdsizemass_detection_{feature1}_{feature2}.pdf'))
    #plt.show()
    plt.close()


# PCA on spot features
from sklearn.decomposition import PCA

featues_spotdetection = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame']

num_eigvectors = 2

# Standardize data
df_spots_norm = df_comparision_filtered[featues_spotdetection]
df_spots_norm = df_spots_norm - df_spots_norm.mean()
df_spots_norm = df_spots_norm / df_spots_norm.std()

# PCA
pca = PCA(n_components=num_eigvectors)
pca.fit(df_spots_norm)
pca_components = pca.transform(df_spots_norm)
pca_components = pd.DataFrame(pca_components, columns=[f'PC{i}' for i in range(1, num_eigvectors + 1)])
pca_components['groundtruth'] = df_comparision_filtered['correct']

print(pca.explained_variance_ratio_)

# Plot
for value, color in colors.items():
    mask = pca_components['groundtruth'] == value
    plt.scatter(pca_components[mask]['PC1'], pca_components[mask]['PC2'], c=color, label=str(value))
plt.xlabel(f'PC1')
plt.ylabel(f'PC2')
plt.legend()
#plt.savefig(os.path.join(path_output, 'PCA_filtered.pdf'))
plt.show()
plt.close()

# combine length of burst for filtering
featues_spotdetection = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']

# First, sort the DataFrame by 'filename', 'track_id', and 'frame'
df_grouped = df_comparision_filtered.sort_values(by=['filename', 'track_id', 'frame'])

# Calculate the difference between consecutive frames within each group
df_grouped['frame_diff'] = df_grouped.groupby(['filename', 'track_id'])['frame'].diff()

# Assign group numbers based on consecutive frames
df_grouped['group_number'] = (df_grouped['frame_diff'] > 1).cumsum()

# Drop the temporary 'frame_diff' column if not needed
df_grouped.drop(columns='frame_diff', inplace=True)

df_grouped_sum = df_grouped.groupby(['filename', 'group_number'])[featues_spotdetection].sum().reset_index()

length_burst = df_grouped.groupby(['filename', 'group_number']).size().reset_index()
df_grouped_sum = df_grouped_sum.merge(length_burst)
df_grouped_sum.columns = [*df_grouped_sum.columns[:-1], 'time_length']
# Define a function to determine if 'correct' column contains only True, only False, or a mix
def check_correctness(group):
    if group['correct'].all():
        return 'only_true'
    elif not group['correct'].any():
        return 'only_false'
    else:
        return 'mix_true_false'

# Apply the function to each group and create a new column 'correctness' in df_grouped_sum
correctness = df_grouped.groupby(['filename', 'group_number']).apply(check_correctness).reset_index()
df_grouped_sum = df_grouped_sum.merge(correctness, on=['filename', 'group_number'])
df_grouped_sum.columns = [*df_grouped_sum.columns[:-1], 'correct']

colors = {'mix_true_false': 'orange', 'only_true': 'black', 'only_false': 'red'}

columns = ['mass', 'signal', 'raw_mass', 'time_length']
unique_combinations = set(combinations(columns, 2))

for feature1, feature2 in unique_combinations:
    for value, color in colors.items():
        mask = df_grouped_sum['correct'] == value
        plt.scatter(df_grouped_sum[mask][feature1], df_grouped_sum[mask][feature2], c=color, label=str(value))
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.legend()
    # plt.savefig(os.path.join(path_output, f'detection_{feature1}_{feature2}.pdf'))
    plt.show()
    plt.close()
