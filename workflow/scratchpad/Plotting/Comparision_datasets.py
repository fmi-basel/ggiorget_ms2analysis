import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#
path_output = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/comparing_plots'

# facs data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/facs/analysis/livecell_fiveclones/seeding_day'
filename = 'ficeclone-liveimaging_gfp-flowcytomery_seeding_summary.csv'
df_facs_five = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})
df_facs_five['dataset'] = 'fiveclones'
df_facs_five['Clone'] = df_facs_five['Clone'].str.replace('E14', 'E14ddCTCF')

path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/facs/analysis/live-imaging/seeding_day'
filename = 'halfenhancer_live-imaging-seeding_gfp-flowcytometry_summary.csv'
df_facs_half = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})
df_facs_half['dataset'] = 'halfenhancer'

path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/facs/analysis/basic_triplicate'
filename = 'halfenhancer_gfp-flowcytometry_summary.csv'
df_facs_halft = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})
df_facs_halft['dataset'] = 'halfenhancer (triplicate, younger cells)'

df_facs = pd.concat([df_facs_five, df_facs_half, df_facs_halft])

# plot mean GFP data of clones as boxplot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_facs[df_facs['Clone'].isin(['5G7', 'E14ddCTCF'])], x='Clone', y='Mean', hue='dataset', ax=ax)
plt.title('Flow cytometry GFP intensity')
plt.ylabel('Mean GFP intensity')
#plt.savefig(os.path.join(path_output, 'facs.pdf'))
plt.show()

# smFISH data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/fish/summarized_data/'
filename = 'spots-per-cell-avg_MS2.csv'
df_fish_five = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})
df_fish_five['dataset'] = 'fiveclones'

path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/fish/summarized_data/'
filename = 'HalfEnhancer_spots-per-cell-avg_MS2.csv'
df_fish_half = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})
df_fish_half['dataset'] = 'halfenhancer'

df_fish = pd.concat([df_fish_five, df_fish_half]).reset_index(drop=True)

# plot mean GFP data of clones as boxplot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_fish[df_fish['clone'].isin(['5G7', 'E14ddCTCF'])], x='clone', y='spots_per_cell', hue='dataset',
            ax=ax)
plt.title('smFISH')
plt.ylabel('average number of spots per cell')
# plt.savefig(os.path.join(path_output, 'fish_avgnumber_boxplot.pdf'))
plt.show()

# count histogramm of spots per cell
grouped = df_fish[df_fish['clone'].isin(['5G7', 'E14ddCTCF'])]
grouped = grouped.groupby(['clone'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
# Flatten the 2D array of subplots to a 1D array
axes = axes.flatten()
# Iterate through groups and plot histograms in separate subplots
for i, (name, group) in enumerate(grouped):
    ax = axes[i]
    sns.histplot(data=group, x='spots_per_cell', binwidth=1, ax=ax, hue='dataset')
    ax.set_title(f'Clone {name}')
    ax.set_xlabel('Spots per Cell')
    ax.set_ylabel('Frequency')
# plt.savefig(os.path.join(path_output, 'fish_hist.pdf'))
plt.show()

# check bleaching in live-cell data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data'
filename = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800.csv'
df_live_five = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})

path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data'
filename = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_HalfEnhancer_combined.csv'
df_live_half = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})

# plot mean_complete cell over time and average over all traces
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df_live_five, x='frame', y='mean_completecell', label='Five clones')
sns.lineplot(data=df_live_half, x='frame', y='mean_completecell', label='Half enhancer')
plt.title('Live-cell: Mean cell intensity over time')
plt.ylabel('Mean intensity')
plt.xlabel('Frame')
plt.legend()
# plt.savefig(os.path.join(path_output, 'cellbackground_over_time.pdf'))
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df_live_five[df_live_five['spotdetected'] == True], x='frame', y='mean_spot', label='Five clones')
sns.lineplot(data=df_live_half[df_live_half['spotdetected'] == True], x='frame', y='mean_spot', label='Half enhancer')
plt.title('Live-cell: Mean spot intensity over time')
plt.ylabel('Mean intensity')
plt.xlabel('Frame')
plt.legend()
#plt.savefig(os.path.join(path_output, 'spotint_over_time.pdf'))
plt.show()

# grountruth over time
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/data/groundtruth_tracks'
filenames = [
    '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
    '20230719_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5E10_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
    '20230720_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5F11_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s5_groundtruth.csv']
df_grundtruth_five = []
for file in filenames:
    df_file = pd.read_csv(os.path.join(path, file))
    df_file['file'] = file
    df_grundtruth_five.append(df_file)
df_grundtruth_five = pd.concat(df_grundtruth_five)
filenames_five = [s.replace('_groundtruth.csv', '_MAX.tiff') for s in filenames]
cellnumber = df_live_five[df_live_five['filename'].isin(filenames_five)].groupby(['frame'])['unique_id'].size().reset_index()
df_grundtruth_five_count = df_grundtruth_five.groupby('frame').size().reset_index()
df_grundtruth_five_count = df_grundtruth_five_count.merge(cellnumber, on='frame', how='left')
df_grundtruth_five_count['norm_count']=df_grundtruth_five_count[0]/df_grundtruth_five_count['unique_id']

path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/groundtruth_halfenh/'
filenames = [
    '20240131_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_4A1_2_FullseqTIRF-mCherry-GFPCy5WithSMB_s3_groundtruth.csv',
    '20240201_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_4A1_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s4_groundtruth.csv',
    '20240201_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_groundtruth.csv',
    '20240203_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_2_FullseqTIRF-mCherry-GFPCy5WithSMB_s2_groundtruth.csv']
df_grundtruth_half = []
for file in filenames:
    df_file = pd.read_csv(os.path.join(path, file))
    df_file['file'] = file
    df_grundtruth_half.append(df_file)
df_grundtruth_half = pd.concat(df_grundtruth_half)
df_grundtruth_half.rename(columns={'axis-0': 'frame'}, inplace=True)
filenames_half = [s.replace('_groundtruth.csv', '_MAX.tiff') for s in filenames]
cellnumber = df_live_half[df_live_half['filename'].isin(filenames_half)].groupby(['frame'])['unique_id'].size().reset_index()
df_grundtruth_half_count = df_grundtruth_half.groupby('frame').size().reset_index()
df_grundtruth_half_count = df_grundtruth_half_count.merge(cellnumber, on='frame', how='left')
df_grundtruth_half_count['norm_count']=df_grundtruth_half_count[0]/df_grundtruth_half_count['unique_id']

# plot number occurances over frame
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9), sharex=True)
# Flatten the 2D array of subplots to a 1D array
axes = axes.flatten()
sns.scatterplot(data=df_grundtruth_five_count, x='frame', y=0, ax=axes[0])
axes[0].set_title('Five clones')
axes[0].set_ylabel('Number of spots')
sns.scatterplot(data=df_grundtruth_half_count, x='frame', y=0, ax=axes[1])
axes[1].set_title('Half enhancer')
axes[1].set_ylabel('Number of spots')
axes[1].set_xlabel('Frame')
plt.tight_layout()
# plt.savefig(os.path.join(path_output, 'groundtruth_over_time.pdf'))
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9), sharex=True)
# Flatten the 2D array of subplots to a 1D array
axes = axes.flatten()
sns.scatterplot(data=df_grundtruth_five_count, x='frame', y='norm_count', ax=axes[0])
axes[0].set_title('Five clones')
axes[0].set_ylabel('Number of spots')
sns.scatterplot(data=df_grundtruth_half_count, x='frame', y='norm_count', ax=axes[1])
axes[1].set_title('Half enhancer')
axes[1].set_ylabel('Number of spots/cell number')
axes[1].set_xlabel('Frame')
plt.tight_layout()
#plt.savefig(os.path.join(path_output, 'groundtruth_over_time_norm.pdf'))
plt.show()
