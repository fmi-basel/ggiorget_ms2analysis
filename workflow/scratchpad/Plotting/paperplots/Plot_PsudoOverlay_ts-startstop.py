"""
In theory, at the start of a burst, the spot intensity should increase over the course of 2-3 frames (MS2 is
transcribed in around 30s, forming more and more loops/getting more intense). In this script I plot the spot intensity
around the burst start to see if I can visualize this.
"""

# Load all the packages
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'size': 6}
matplotlib.rc('font', **font)


def indicate_first_last_series(x, column):
    """
    From a series, indicate the if the occurrence is left-, right-, non-censored or both, right and left censored.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
    Returns:
         data: (pd.Dataframe) processed dataframe with an additional column containing the censoring information
    """
    data = x.copy()
    # indicate groups of values
    data = data.assign(group_no=(data[column] != data[column].shift()).cumsum())
    min_id = data['group_no'].min()
    max_id = data['group_no'].max()
    data['censored'] = 'noncensored'
    data.loc[data['group_no'] == min_id, 'censored'] = 'leftcensored'
    data.loc[data['group_no'] == max_id, 'censored'] = 'rightcensored'
    if min_id == max_id:
        data['censored'] = 'nonbursting'
    data.drop(columns='group_no', inplace=True)
    return data


def overlap_traces_start_average(dataframe, binerized_trace, intensity_trace):
    # get indexes meeting the criterions for overlap
    criteria = dataframe[(dataframe[binerized_trace] == True) & (dataframe['signal_change'] == True)]
    criteria_index = criteria.index
    # make dataframe with all traces
    df_overlay_all = []
    for index in criteria_index:
        selection = dataframe.loc[index - 9:index + 15].reset_index(drop=True)
        cell_id = selection.loc[9, 'unique_id']
        selection.loc[selection['unique_id']!=cell_id, intensity_trace] = np.nan
        selection['group_no'] = (selection[binerized_trace] != selection[binerized_trace].shift()).cumsum()
        groupno_burst = selection['group_no'].iloc[9]
        selection.loc[~selection['group_no'].isin(range(groupno_burst-1,groupno_burst+2)), intensity_trace] = np.nan
        df_overlay_all.append(selection[intensity_trace])
    df_overlay_all = pd.DataFrame(df_overlay_all).transpose()
    # calculate mean and std of the traces
    mean_series = df_overlay_all.mean(axis=1, skipna=True).rename('mean')
    std_series = df_overlay_all.std(axis=1, skipna=True).rename('std')
    return pd.concat([mean_series, std_series], axis=1)


def overlap_traces_start_derivative(dataframe, binerized_trace, intensity_trace):
    # get indexes meeting the criterions for overlap
    criteria = dataframe[(dataframe[binerized_trace] == True) & (dataframe['signal_change'] == True)]
    criteria_index = criteria.index
    # make dataframe with all traces
    df_overlay_all = []
    for index in criteria_index:
        selection = dataframe.loc[index - 9:index + 15].reset_index(drop=True)
        cell_id = selection.loc[9, 'unique_id']
        selection.loc[selection['unique_id'] != cell_id, intensity_trace] = np.nan
        selection['group_no'] = (selection[binerized_trace] != selection[binerized_trace].shift()).cumsum()
        groupno_burst = selection['group_no'].iloc[9]
        selection.loc[
            ~selection['group_no'].isin(range(groupno_burst - 1, groupno_burst + 2)), intensity_trace] = np.nan
        df_overlay_all.append(selection[intensity_trace])
    df_overlay_all = pd.DataFrame(df_overlay_all).transpose()
    df_overlay_all_derivatives = df_overlay_all.diff()

    mean_series = df_overlay_all_derivatives.mean(axis=1).rename('mean')
    std_series = df_overlay_all_derivatives.std(axis=1).rename('std')

    return pd.concat([mean_series, std_series], axis=1)


mm = 1 / 25.4
green_5e10 = (111 / 255, 188 / 255, 133 / 255)
blue_5f11 = (113 / 255, 171 / 255, 221 / 255)
purple_5g3 = (156 / 255, 107 / 255, 184 / 255)
red_5g7 = (213 / 255, 87 / 255, 69 / 255)
colors = [green_5e10, blue_5f11, purple_5g3, red_5g7]

# Load data
path_out = '/Users/janatunnermann/Desktop/plots_paper/'
data_path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800_curated.csv'
df = pd.read_csv(data_path, low_memory=False)

# which datatype to do the calculations with
datatype = 'spotdetected_filtered_curated'
intensity_read_out = 'corr_trace'

# ---- Pre-processing ----
df = df[df['frame'] >= 120]
df = df[df['frame'] <= 600]
df = df[df[datatype].notna()].reset_index(drop=True)

# indicate the start-time of a signal change (on/off and off/on)
df = df.sort_values(['unique_id', 'frame']).reset_index(drop=True)
df['signal_change'] = df.groupby('unique_id').apply(lambda x: x[datatype] != x[datatype].shift()).reset_index(drop=True)
# indicate left and right censored data
df = df.groupby(['unique_id'], group_keys=True).apply(indicate_first_last_series, column=datatype).reset_index(
    drop=True)

# exclude non bursting cells and left-censored bursts
df = df[df['censored'] != 'nonbursting']
df = df[~((df['censored'] == 'leftcensored') & (df[datatype] == True))].reset_index(drop=True)
df = df[~((df['censored'] == 'rightcensored') & (df[datatype] == True))].reset_index(drop=True)

# -- Intensity trace --
# Group the DataFrame by the 'group' column
grouped = df.groupby('clone')
# Apply your function to each group and filter based on the condition
df_overlay_grouped_meanint = []
for name, group in grouped:
    processed_group = overlap_traces_start_average(group, binerized_trace=datatype, intensity_trace=intensity_read_out)
    processed_group['clone'] = name
    df_overlay_grouped_meanint.append(processed_group)
df_overlay_grouped_meanint = pd.concat(df_overlay_grouped_meanint)
clones = df_overlay_grouped_meanint['clone'].unique()

fig, ax = plt.subplots(figsize=(35*mm, 27*mm))
for i, clone in enumerate(clones):
    color = colors[i]
    data = df_overlay_grouped_meanint[df_overlay_grouped_meanint['clone'] == clone]
    ax.plot((data.index - 9) / 2, data['mean'], color=color, label=f'{clone}', linewidth=1)
    ax.fill_between((data.index - 9) / 2, data['mean'] - data['std'],
                    data['mean'] + data['std'], color=color, alpha=0.2)
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Time (min)')
x_values = np.arange(int(min(data.index - 9) / 2), int(max(data.index - 9) / 2) + 1, 2)
ax.set_xticks(x_values)
ax.set_xticklabels(x_values)
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.1)
limits = ax.get_xlim()
plt.savefig(os.path.join(path_out, 'Intensity-trace_overlay.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# -- Derivative intensity trace --
grouped = df.groupby('clone')
# Apply your function to each group and filter based on the condition
df_overlay_grouped_drivative = []
for name, group in grouped:
    processed_group = overlap_traces_start_derivative(group, binerized_trace=datatype,
                                                      intensity_trace=intensity_read_out)
    processed_group['clone'] = name
    df_overlay_grouped_drivative.append(processed_group)
df_overlay_grouped_drivative = pd.concat(df_overlay_grouped_drivative)


fig, ax = plt.subplots(figsize=(45*mm, 39*mm))
for i, clone in enumerate(clones):
    color = colors[i]
    data = df_overlay_grouped_drivative[df_overlay_grouped_drivative['clone'] == clone]
    ax.plot((data.index - 9) / 2, data['mean'], color=color, label=f'{clone}')
    ax.fill_between((data.index - 9) / 2, data['mean'] - data['std'],
                    data['mean'] + data['std'], color=color, alpha=0.2)
x_values = np.arange(int(min(data.index - 9) / 2), int(max(data.index - 9) / 2) + 1, 2)
ax.set_xticks(x_values)
ax.set_xticklabels(x_values)
plt.xlim(limits)
plt.ylabel('Intensity derivative (a.u.)')
plt.xlabel('Time (min)')
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, 'Intensity-trace_overlay_derivative.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()
