# Load all the packages
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from skimage.morphology import remove_small_holes

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'size': 6}
matplotlib.rc('font', **font)
import matplotlib.ticker as ticker

mm = 1 / 25.4

def signallength(x, column):
    """
    From a series calculate how long a state was present, including information about GFP levels.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
    Returns:
         burst_stats: (pd.Dataframe) processed dataframe
    """
    x = x.assign(signal_no=(x[column] != x[column].shift()).cumsum())
    burst_stats = x.groupby(['signal_no', column]).agg(
        {'frame': ['count'], 'corr_trace': ['sum', 'max']}).reset_index()
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
path_out = '/Users/janatunnermann/Desktop/plots_paper'
path_annotated_data = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/annotated_tracks/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_manualannotated_combined.csv'
path_pipeline_data = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800_curated.csv'

df_annotated = pd.read_csv(path_annotated_data)
df_annotated['detectiontype'] = 'annotated'
df_annotated = df_annotated[df_annotated['clone'] != '5E10']
df_annotated['spotdetected_filtered_curated'] = df_annotated.groupby('unique_id').apply(
    lambda cell: remove_positives(cell.spotdetected, area_size=2)).reset_index(level=0, drop=True)

df_pipeline = pd.read_csv(path_pipeline_data, low_memory=False)
df_pipeline['detectiontype'] = 'pipeline'

df_pipeline_sub = df_pipeline[df_pipeline['filename'].isin(df_annotated['filename'].unique())]
df_pipeline_sub['detectiontype'] = 'pipeline_subset'

df = pd.concat([df_annotated, df_pipeline_sub, df_pipeline], ignore_index=True)

# some preprocessing
datatype = 'spotdetected_filtered_curated'

df = df[df['frame'] >= 120]
df = df[df['frame'] <= 600]
df = df[df['spotdetected_filtered_curated'].notna()]

clones = np.unique(df['clone'])
clones = np.roll(clones, 1)
clones_pretty = clones[1:5]
green_5e10 = (111 / 255, 188 / 255, 133 / 255)
blue_5f11 = (113 / 255, 171 / 255, 221 / 255)
purple_5g3 = (156 / 255, 107 / 255, 184 / 255)
red_5g7 = (213 / 255, 87 / 255, 69 / 255)
colors_gregory = [green_5e10, blue_5f11, purple_5g3, red_5g7]

# ---- Summarizing statistics ----
# sum-up length of states
df_signallength = df.groupby(['unique_id', 'detectiontype', 'clone'], group_keys=True).apply(signallength,
                                                                                             column=datatype).reset_index()
# Indicate which values are incomplete/censored
df_signallength = df_signallength.groupby(['unique_id', 'detectiontype', 'clone'], group_keys=True).apply(
    indicate_first_last, column='signal_no').reset_index(drop=True)

# ---- KM ----
# drop left censored and non-bursting
df_km = df_signallength[df_signallength.censored != 'leftcensored']
df_km = df_km[df_km.censored != 'nonbursting']
# keep right censored and non-censored
df_km.loc[df_km['censored'] == 'rightcensored', 'censored'] = 0
df_km.loc[df_km['censored'] == 'noncensored', 'censored'] = 1

# Fitting the Kaplan-Meier
# chose off or on times
signal = False
df_km_times = df_km[df_km[datatype] == signal]

kmf = KaplanMeierFitter()

# compare annotated and pipeline
clones = ['5G7', '5F11']
signal_type = 'on' if signal else 'off'

fig, axs = plt.subplots(2, 1, figsize=(44*mm, 2 * 1.5), sharey=True, sharex=True)
for i, ax in enumerate(axs.ravel()):
    clone = clones[i]
    color_clone = colors_gregory[3] if clone == '5G7' else colors_gregory[1]
    color_id = [color_clone, 'dimgrey', 'darkgrey']
    for j, detection_type in enumerate(['pipeline', 'pipeline_subset', 'annotated']):
        kmf.fit((df_km_times[(df_km_times['detectiontype'] == detection_type) & (df_km_times['clone'] == clone)][
                     'framecount'] / 2),
                event_observed=
                df_km_times[(df_km_times['detectiontype'] == detection_type) & (df_km_times['clone'] == clone)][
                    'censored'])
        km_times = kmf.survival_function_.index.values
        km_survival_probs = kmf.survival_function_.values
        km_survival_probs[km_survival_probs == 0] = np.nan
        km_ci_lower = kmf.confidence_interval_.iloc[:, 0].values
        km_ci_lower[km_ci_lower == 0] = np.nan
        km_ci_upper = kmf.confidence_interval_.iloc[:, 1].values
        km_ci_upper[km_ci_upper == 0] = np.nan
        ax.plot(km_times[1:], km_survival_probs[1:], color=color_id[j], label=detection_type)
        ax.fill_between(km_times[1:], km_ci_lower[1:], km_ci_upper[1:], color=color_id[j], alpha=0.2)
    ax.tick_params('x', labelbottom=True)
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Time (min)')
    ax.grid(alpha=0.2)
    ax.tick_params(axis='both', which='both', length=1.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    # ax.legend()
    # limit = ax.get_ylim()
# plt.ylim(bottom=limit[0])
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(path_out, f'{signal_type}times_comparision.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# --- Burst size or amplitude---
# drop left censored and non-bursting
df_sizeamp = df_signallength[df_signallength.censored != 'leftcensored']
df_sizeamp = df_sizeamp[df_sizeamp.censored != 'rightcensored']

# Comparision
# define what metric to use (sum (burst size) or max intensity(burst amplitude)
metric = 'corr_tracesum'  # corr_tracemax or corr_tracesum
clones = ['5G7', '5F11']
detection_type = ['pipeline', 'pipeline_subset', 'annotated']
metric_type = 'size' if 'sum' in metric else 'amplitude'

# calculate confidence interval using bootstrapping
confidence_interval = []
for clone, detectiontype in itertools.product(clones, detection_type):
    # calculate cdf and survival function
    cdf = df_sizeamp[
        (df_sizeamp[datatype] == True) & (df_sizeamp['clone'] == clone) & (
                    df_sizeamp['detectiontype'] == detectiontype)][
        metric].value_counts(
        normalize=True).sort_index().cumsum()
    cdf.reset_index()
    survival = 1 + cdf.iloc[0] - cdf

    # Number of bootstrap samples
    num_samples = 10000  # You can adjust this as needed

    # Create an array to store the bootstrapped CDF values
    bootstrap_survivals = np.zeros((num_samples, len(survival)))
    # Perform bootstrapping
    for i in range(num_samples):
        # Resample with replacement from your original data
        bootstrap_sample = np.random.choice(survival.index, size=len(survival), replace=True)

        # Calculate the CDF of the bootstrap sample
        bootstrap_survival = pd.Series(bootstrap_sample).value_counts(normalize=True).sort_index().cumsum()
        bootstrap_survival = 1 + bootstrap_survival.iloc[0] - bootstrap_survival

        # Interpolate the bootstrap CDF to match the original CDF indices
        bootstrap_survival = np.interp(survival.index, bootstrap_survival.index, bootstrap_survival.values)
        bootstrap_survivals[i, :] = bootstrap_survival

    # Calculate the confidence intervals
    alpha = 0.05  # Confidence level (e.g., 95% confidence interval)
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_survivals, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_survivals, upper_percentile, axis=0)

    df = pd.DataFrame({'lower_bound': lower_bound, 'upper_bound': upper_bound})
    df['clone'] = clone
    df['detection_type'] = detectiontype
    confidence_interval.append(df)
confidence_interval = pd.concat(confidence_interval)

# plot
fig, axs = plt.subplots(2, 1, figsize=(44*mm, 2 * 1.5), sharey=True, sharex=True)
for i, ax in enumerate(axs.ravel()):
    clone = clones[i]
    color_clone = colors_gregory[3] if clone == '5G7' else colors_gregory[1]
    color_id = [color_clone, 'dimgrey', 'darkgrey']
    for j, detection_type in enumerate(['pipeline', 'pipeline_subset', 'annotated']):
        cdf = df_sizeamp[
            (df_sizeamp[datatype] == True) & (df_sizeamp['clone'] == clone) & (
                        df_sizeamp['detectiontype'] == detection_type)][
            metric].value_counts(normalize=True).sort_index().cumsum()
        survival = 1 + cdf.iloc[0] - cdf
        ax.plot(survival, linewidth=1, label=detection_type, color=color_id[j])
        ax.fill_between(survival.index, confidence_interval[
            (confidence_interval['clone'] == clone) & (confidence_interval['detection_type'] == detection_type)][
            'lower_bound'],
                        confidence_interval[(confidence_interval['clone'] == clone) & (
                                    confidence_interval['detection_type'] == detection_type)]['upper_bound'], alpha=0.2,
                        color=color_id[j])
    ax.tick_params('x', labelbottom=True)
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Intensity (a.u.)')
    ax.grid(alpha=0.2)
    ax.tick_params(axis='both', which='both', length=1.5)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    # ax.legend()
plt.tight_layout(pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig(os.path.join(path_out, f'burst{metric_type}_comparision.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()
