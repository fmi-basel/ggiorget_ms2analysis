# Load all the packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib
from skimage.morphology import remove_small_holes
import itertools
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'size': 16}
matplotlib.rc('font', **font)
import matplotlib.ticker as ticker

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
    burst_stats = x.groupby(['signal_no', column]).agg(
        {'frame': ['count']}).reset_index()
    burst_stats.columns = list(map(''.join, burst_stats.columns.values))
    return burst_stats


def indicate_first_last(x, column):
    """
    From a series, indicate the if the occurrence is left-, right-, non-censored or both, right and left censored.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
    Returns:
         data: (pd.Dataframe) processed dataframe with an additional column containing the censoring information
    """
    data = x.copy()
    data['censored'] = 'noncensored'
    min_id = data[column].min()
    max_id = data[column].max()
    number_id = data[column].sum()
    data.loc[data[column] == min_id, 'censored'] = 'leftcensored'
    data.loc[data[column] == max_id, 'censored'] = 'rightcensored'
    if min_id == max_id:
        data['censored'] = 'nonbursting'
    return data


def remove_positives(data, area_size=2):
    """
    From a bool signal trace, removes True values of certain length
    Args:
         data: (pd.Series) data trace to be processed
         area_size: (int) minimal length of consecutive True values to keep

    Returns:
         data_array_filtered: (pd.Series) filtered data trace
    """
    data_array = data.apply(lambda x: not x)
    data_array_filtered = remove_small_holes(data_array, area_threshold=area_size)
    data_array_filtered = ~data_array_filtered.astype(bool)
    return data_array_filtered


# Load data
path_out = '/Users/janatunnermann/Desktop/'
path_annotated_data = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/annotated_tracks/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_manualannotated_combined.csv'
path_pipeline_data = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800_curated.csv'

df_annotated = pd.read_csv(path_annotated_data)
df_annotated['detectiontype'] = 'annotated'
#df_annotated['spotdetected_curated'] = df_annotated['spotdetected']
df_annotated['spotdetected_filtered_curated'] = df_annotated.groupby('unique_id').apply(lambda cell: remove_positives(cell.spotdetected, area_size=2)).reset_index(level=0, drop=True)
#df_annotated['spotdetected_curated_filtered'] = df_annotated['spotdetected_curated']

df_pipeline = pd.read_csv(path_pipeline_data, low_memory=False)
df_pipeline['spotdetected_filtered_curated'] = df_pipeline['spotdetected_filtered']
df_pipeline.loc[df_pipeline['spotdetected_curated'] == np.nan,'spotdetected_filtered_curated'] = np.nan
df_pipeline.loc[df_pipeline['clone'] == '6G3','spotdetected_filtered_curated'] = False
df_pipeline['detectiontype'] = 'pipeline'
df = df_pipeline

df_pipeline_sub = df_pipeline[df_pipeline['filename'].isin(df_annotated['filename'].unique())]
df_pipeline_sub['detectiontype'] = 'pipeline_subset'

df_pipeline = df_pipeline[(df_pipeline['clone']=='5G7')|(df_pipeline['clone']=='5F11')]
df_pipeline['detectiontype'] = 'pipeline'

df = pd.concat([df_annotated, df_pipeline_sub, df_pipeline], ignore_index=True)

# some preprocessing
datatype = 'spotdetected_filtered_curated'

#df = df_annotated.copy()
df = df[df['frame']>120]
df = df[df['frame']<600]

# ---- Summarizing statistics ----
# sum-up length of states
df_signallength = df.groupby(['unique_id', 'detectiontype', 'clone']).apply(signallength, column=datatype).reset_index()

# Indicate which values are incomplete/censored
df_signallength = df_signallength.groupby(['unique_id', 'detectiontype', 'clone']).apply(indicate_first_last, column='signal_no').reset_index()

# decide on which data to use
# drop left censored
df_signallength = df_signallength[df_signallength.censored != 'leftcensored']
# keep right censored and non-censored
df_signallength.loc[df_signallength['censored'] == 'rightcensored', 'censored'] = 0
df_signallength.loc[df_signallength['censored'] == 'noncensored', 'censored'] = 1
# Either drop or keep non bursting
df_signallength = df_signallength[df_signallength.censored != 'nonbursting']
#df_signallength.loc[df_signallength['censored'] == 'nonbursting', 'censored'] = 0

# ---- Fitting ----
signal = False
# chose off times with at least one burst in trace
df_signallength_times = df_signallength[df_signallength[datatype] == signal]
# df_signallength_times = df_signallength[(df_signallength[datatype]==signal) & (df_signallength['framecount']<361)]

#df_signallength_times.drop(columns=['level_2', 'index', 'detectiontype'], inplace=True)
#df_signallength_times.to_csv(os.path.join(path_out, 'censored_offtimes_annotateddata.csv'), index=False)

kmf = KaplanMeierFitter()
'''
kmf.fit(df_signallength_times['framecount']/2, event_observed=df_signallength_times['censored'])
plotvalues = kmf.survival_function_
plotvalues = plotvalues.reset_index()
plotvalues.to_csv(os.path.join(path_out, 'km_surv.csv'), index=False)
'''

fig, ax = plt.subplots(figsize=(8, 6))
#for combination in itertools.product(df_signallength_times['detectiontype'].unique(), df_signallength_times['clone'].unique()):
for combination in itertools.product(df_signallength_times['detectiontype'].unique(), np.array(['5F11'])):
    sample, clone = combination
    kmf.fit((df_signallength_times[(df_signallength_times['detectiontype'] == sample) & (df_signallength_times['clone'] == clone)]['framecount']/2),
            event_observed=df_signallength_times[(df_signallength_times['detectiontype'] == sample) & (df_signallength_times['clone'] == clone)]['censored'],
            label=sample)
    kmf.plot(ax=ax)
plt.yscale('log')
plt.xlabel('Time [min]')
plt.ylabel('Survival probability')
plt.title(f'K-M on-times, {clone}')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tight_layout()
#plt.ylim(0.0001,1.05)
#plt.xlim(-1, 9)
#plt.savefig(os.path.join(path_out, f'ontimes_annotateddata_{clone}_filtered.pdf'))
plt.show()
plt.close()


clone = '5G7'
fig, ax = plt.subplots(figsize=(8, 6))
kmf.fit((df_signallength_times[(df_signallength_times['detectiontype'] == 'annotated') & (df_signallength_times['clone'] == clone)]['framecount']/2),
            event_observed=df_signallength_times[(df_signallength_times['detectiontype'] == 'annotated') & (df_signallength_times['clone'] == clone)]['censored'],
            label='annotated')
kmf.plot(ax=ax)
kmf.fit((df_signallength_times[(df_signallength_times['detectiontype'] == 'pipeline_subset') & (df_signallength_times['clone'] == clone)]['framecount']/2),
            event_observed=df_signallength_times[(df_signallength_times['detectiontype'] == 'pipeline_subset') & (df_signallength_times['clone'] == clone)]['censored'],
            label='spotdetected_curated_filtered')
kmf.plot(ax=ax)
plt.yscale('log')
plt.xlabel('Time [min]')
plt.ylabel('Survival probability')
#plt.title(f'K-M on-times, {clone}')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tight_layout()
#plt.ylim(0.0001,1.05)
#plt.xlim(-1, 9)
#plt.savefig(os.path.join(path_out, f'ontimes_annotateddata_{clone}_filtered.pdf'))
plt.show()
plt.close()


# ---- Singlets ----
df_singlets = df[(df['detectiontype'] == 'annotated') | (df['detectiontype'] == 'pipeline_subset')][['track_id', 'filename', 'frame', 'spotdetected', 'spotdetected_filtered_curated', 'detectiontype', 'clone']]
df_singlets['singlets'] = (df_singlets['spotdetected_filtered_curated'] != df_singlets['spotdetected']).astype(int)

# mutate this df so that based on detection type columns matching in track_id and filename  are placed in the same row
df_singlets = df_singlets.pivot_table(index=['track_id', 'filename', 'frame', 'clone'], columns='detectiontype', values='singlets').reset_index()
df_singlets['removed'] = df_singlets['annotated'] - df_singlets['pipeline_subset']


fig, ax = plt.subplots(figsize=(8, 6))
#for combination in itertools.product(df_signallength_times['detectiontype'].unique(), df_signallength_times['clone'].unique()):
for clone in df_signallength_times['clone'].unique():
    kmf.fit((df_signallength_times[df_signallength_times['clone'] == clone]['framecount'] / 2),
            event_observed=df_signallength_times[df_signallength_times['clone'] == clone]['censored'],
            label=clone)
    kmf.plot(ax=ax)
plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Time [min]')
plt.ylabel('Survival probability')
plt.title(f'K-M off-times')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tight_layout()
#plt.ylim(0.0001,1.05)
#plt.xlim(-1, 9)
plt.savefig(os.path.join(path_out, f'offtimes_filtered_log.pdf'))
plt.show()
plt.close()
