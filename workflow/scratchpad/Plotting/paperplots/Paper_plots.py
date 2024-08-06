import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'size': 6}
matplotlib.rc('font', **font)
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy import stats


def remove_zero_text(x):
    if len(x) >= 3 and x[2] == '0':
        return x[:2] + x[3:]
    else:
        return x


def exclude_first_last(x, column, first=True, last=True):
    """
    From a series (numbers, bool etc.), set the first or last occurrence to nan
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
         first: (bool) exclude first occurrence
         last: (bool) exclude last occurrence
    Returns:
         data: (pd.Dataframe) processed dataframe
    """
    import numpy as np
    data = x.copy()
    data = data.assign(exclude=(data[column] != data[column].shift()).cumsum())
    if first:
        min_id = data['exclude'].min()
        data.loc[data['exclude'] == min_id, column] = np.nan
    if last:
        max_id = data['exclude'].max()
        data.loc[data['exclude'] == max_id, column] = np.nan
    data.drop(columns=['exclude'], inplace=True)
    return data


def signallength_includingintensities2(x, column):
    """
    From a series calculate how long a state was present, including information about GFP levels.
    Args:
         x: (pd.Dataframe) dataframe in which one column should be processed
         column: (str) column name from x to be processed
    Returns:
         burst_stats: (pd.Dataframe) processed dataframe
    """
    x = x.assign(signal_no=(x[column] != x[column].shift()).cumsum())
    burst_stats = x.groupby(['unique_id', 'signal_no', column]).agg(
        {'frame': ['count'], 'corr_trace': ['sum', 'max'],
         'integrated_intensity_gfp': ['sum', 'max']}).reset_index()
    burst_stats.columns = list(map(''.join, burst_stats.columns.values))
    return burst_stats


path_out = '/Users/janatunnermann/Desktop/plots_paper/'

# --- Load data ---
FACS_data_plate = path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/facs/analysis/MobiPlates5.6/analysis/SummaryFACS_Mobiplate56.csv'
df_facs_plate = pd.read_csv(FACS_data_plate)
df_facs_plate['clone'] = df_facs_plate['clone'].apply(remove_zero_text)
df_facs_plate.set_index('clone', inplace=True)

FACS_data_exp = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/facs/analysis/livecell_fiveclones/imaging_day/ficeclone-liveimaging_gfp-flowcytomery_summary.csv'
df_facs_exp = pd.read_csv(FACS_data_exp, dtype={'Clone': 'str'})

FISH_data = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/fish/summarized_data/spots-per-cell-avg_MS2.csv'
df_fish = pd.read_csv(FISH_data)

trendline = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/trendline_zuinetal.csv'
df_trendline = pd.read_csv(trendline, header=None)

live_imaging = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800_curated.csv'
df_burst = pd.read_csv(live_imaging, dtype={'clone': 'str'}, low_memory=False)
df_burst = df_burst[df_burst['frame'] >= 120]
df_burst = df_burst[df_burst['frame'] <= 600]
df_burst = df_burst[df_burst['spotdetected_filtered_curated'].notna()]
datatype = 'spotdetected_filtered_curated'
bursting = df_burst.groupby(['unique_id']).apply(lambda x: x[datatype].any()).reset_index()
bursting.columns = ['unique_id', 'bursting']
df_burst = df_burst.merge(bursting, on='unique_id')
df_burst.groupby('clone')['unique_id'].nunique()

positions = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/Splinkerette/summary_output/Positions_plate5.6_withcontactprob.csv'
#positions = '/Users/janatunnermann/Desktop/prettyplots/Positions_plate5.6.csv'
df_positions = pd.read_csv(positions)
df_positions['clone'] = df_positions['clone'].apply(remove_zero_text)
df_positions.rename(columns={'cp_distance': 'cp'}, inplace=True)

# --- add cp and position info ---
df_burst = df_burst.merge(df_positions[['clone', 'distance', 'cp']], on='clone')
df_facs_plate = df_facs_plate.merge(df_positions[['clone', 'distance', 'cp']], on='clone')

# General settings
# define colors as for Gregorys plots
clones = np.unique(df_burst['clone'])
clones = np.roll(clones, 1)
clones_pretty = clones[1:5]
green_5e10 = (111 / 255, 188 / 255, 133 / 255)
blue_5f11 = (113 / 255, 171 / 255, 221 / 255)
purple_5g3 = (156 / 255, 107 / 255, 184 / 255)
red_5g7 = (213 / 255, 87 / 255, 69 / 255)
colors_gregory = [green_5e10, blue_5f11, purple_5g3, red_5g7]


# --- Flow cytometry with Jessica's trend-line ---
# scale trend-line to min and max value from my experiment (maybe fit instead?)
df_trendline_scaled = df_trendline.copy()
# Define the minimum and maximum values for scaling
min_value, min_cp = df_facs_plate.loc[df_facs_plate['clone'] == '6G3', ['Mean', 'cp']].values[0]
max_cp = 1
max_value = df_facs_plate.loc[df_facs_plate['cp'] == max_cp, ['Mean']].mean().values[0]
# Extract the original y values corresponding to x1 and x2
min_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0, df_trendline_scaled.columns[1]].values[0]
max_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0.99, df_trendline_scaled.columns[1]].values[0]

# Calculate slope (m) and intercept (b)
m = (max_value - min_value) / (max_original - min_original)
b = min_value - m * min_original

# Apply the linear transformation to the entire y column
df_trendline_scaled['y_scaled'] = m * df_trendline_scaled.iloc[:, 1] + b

plt.figure(figsize=(1.57, 1))
plt.errorbar(df_facs_plate['cp'], df_facs_plate['Mean'], yerr=df_facs_plate['SD'], fmt="o", color='black', markersize=1,
             capsize=1,          # Size of the error bar caps
             capthick=0.5,         # Thickness of the error bar caps
             elinewidth=0.5)
plt.plot(df_trendline_scaled[0], df_trendline_scaled['y_scaled'], color='purple', alpha=0.6, linewidth=1)
plt.grid(alpha=0.2)
x_ticks = np.linspace(0, 1, 5)
plt.xticks(x_ticks)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Contact Probability')
plt.ylabel('Mean eGFP\nintensity (a.u.)')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, 'GFP_facs_trendline.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# --- Correlation Flow cytometry: plate and expanded clones and experiment ---
# calculate stats
df_facs_exp_stat = df_facs_exp.groupby('Clone').describe()

df_facs_plate.set_index('clone', inplace=True)
common_index = df_facs_plate.index.intersection(df_facs_exp_stat.index)
df_facs_plate_comp = df_facs_plate.loc[common_index]
df_facs_exp_stat_comp = df_facs_exp_stat.loc[common_index]
r2 = stats.pearsonr(df_facs_plate_comp['Mean'], df_facs_exp_stat_comp['Mean', 'mean'])[0] ** 2

fig, ax = plt.subplots()
ax.errorbar(df_facs_plate_comp['Mean'], df_facs_exp_stat_comp['Mean', 'mean'], xerr=df_facs_plate_comp['SD'],
            yerr=df_facs_exp_stat_comp['Mean', 'std'], fmt="o", color='black')
sns.regplot(df_facs_plate_comp['Mean'], df_facs_exp_stat_comp['Mean', 'mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}')
plt.xlabel('Mean plate (a.u.)')
plt.ylabel('Mean expanded clones (a.u.)')
# plt.title('Correlation flow cytometry - plate vs expanded')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'GFP_Correlation-facs-plateexpanded.pdf'))
plt.show()
plt.close()

df_gfp_live = df_burst[['unique_id', 'clone', 'mean_gfp']].drop_duplicates().dropna().groupby('clone')['mean_gfp'].describe()
common_index = df_gfp_live.index.intersection(df_facs_exp_stat.index)
df_gfp_live_comp = df_gfp_live.loc[common_index]
df_facs_exp_stat_comp = df_facs_exp_stat.loc[common_index]
r2 = stats.pearsonr(df_gfp_live_comp['mean'], df_facs_exp_stat_comp['Mean', 'mean'])[0] ** 2

fig, ax = plt.subplots()
#ax.errorbar(df_gfp_live_comp['mean'], df_facs_exp_stat_comp['Mean', 'mean'], xerr=df_gfp_live_comp['std'],
#            yerr=df_facs_exp_stat_comp['Mean', 'std'], fmt="o", color='black')
sns.regplot(df_gfp_live_comp['mean'], df_facs_exp_stat_comp['Mean', 'mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}')
plt.xlabel('Mean GFP from microscopy (a.u.)')
plt.ylabel('Mean GFP from flow cytometry (a.u.)')
# plt.title('Correlation flow cytometry - plate vs expanded')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'GFP_Correlation-facs-microscopy.pdf'))
plt.show()
plt.close()


common_index = df_gfp_live.index.intersection(df_facs_plate.index)
df_gfp_live_comp = df_gfp_live.loc[common_index]
df_facs_plate_comp = df_facs_plate.loc[common_index]
r2 = stats.pearsonr(df_gfp_live_comp['mean'], df_facs_plate_comp['Mean'])[0] ** 2

fig, ax = plt.subplots(figsize=(2, 2))
#ax.errorbar(df_gfp_live_comp['mean'], df_facs_plate_comp['Mean'], xerr=df_gfp_live_comp['std'],
#            yerr=df_facs_exp_stat_comp['Mean', 'std'], fmt="o", color='black')
sns.regplot(df_gfp_live_comp['mean'], df_facs_plate_comp['Mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}', scatter_kws={'s': 10, 'linewidths':0}, line_kws={'linewidth': 1})
plt.xlabel('Mean GFP from microscopy (a.u.)')
plt.ylabel('Mean GFP from flow cytometry (a.u.)')
# plt.title('Correlation flow cytometry - plate vs expanded')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'GFP_Correlation-facsplate-microscopy.pdf'), transparent=True)
plt.show()
plt.close()

# --- Track length distribution ---
for clone in df_burst['clone'].unique():
    plt.hist(df_burst[df_burst['clone'] == clone].groupby(['unique_id']).size(), bins=60, label=f'{clone}', alpha=0.5, density=True)
plt.xlabel('Track length (frames)')
plt.ylabel('count')
plt.legend()
plt.title('Track length distribution')
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'track_length_distribution_clone.pdf'))
plt.show()
plt.close()


# --- percentage of cells bursting as cp/distance ---
# calculate percentage of cells bursting from full tracks only
percentage_df = df_burst.groupby('unique_id').filter(lambda x: x.frame.count() >= 480)
percentage_df = percentage_df.groupby(['unique_id']).first().groupby('clone')['bursting'].value_counts().unstack(
    fill_value=0)
percentage_df = percentage_df.div(percentage_df.sum(axis=1), axis=0) * 100
percentage_df = percentage_df.merge(df_positions[['clone', 'distance', 'cp']], on='clone')
percentage_df.set_index('clone', inplace=True)

df_trendline_scaled = df_trendline.copy()
# Define the minimum and maximum values for scaling
min_value, min_cp = percentage_df.loc['6G3', [True, 'cp']].values
max_value, max_cp = percentage_df.loc['5G7', [True, 'cp']].values
# Extract the original y values corresponding to x1 and x2
min_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == round(min_cp,2), df_trendline_scaled.columns[1]].values[0]
max_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0.99, df_trendline_scaled.columns[1]].values[0]

# Calculate slope (m) and intercept (b)
m = (max_value - min_value) / (max_original - min_original)
b = min_value - m * min_original

# Apply the linear transformation to the entire y column
df_trendline_scaled['y_scaled'] = m * df_trendline_scaled.iloc[:, 1] + b
#percentage of bursting
plt.figure(figsize=(1.57, 1))
plt.scatter(percentage_df['cp'], percentage_df[True], color='black', zorder=3, s=3)
plt.plot(df_trendline_scaled[0], df_trendline_scaled['y_scaled'], color='purple', linewidth=1, alpha=0.6, zorder=2)
plt.grid(alpha=0.2, zorder=1)
x_ticks = np.linspace(0, 1, 5)
plt.xticks(x_ticks)
plt.tick_params(axis='both', which='both', length=1.5)
plt.xlabel('Contact probability')
plt.ylabel('% of cells bursting')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, 'Fig1', 'percentage_bursting_long_tracks_trendline.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# use all tracks and normalize for track length -> burst probability
df_signallength = df_burst.groupby(['unique_id']).apply(signallength_includingintensities2, column=datatype).reset_index(drop=True)
df_signallength['clone'] = df_signallength['unique_id'].map(df_burst.groupby(['unique_id'])['clone'].first())
df_signallength['bursting'] = df_signallength['unique_id'].map(df_burst.groupby(['unique_id'])['bursting'].first())

df_signalcount = df_signallength[['unique_id', datatype]].groupby('unique_id').value_counts().reset_index()
df_signalcount.columns = ['unique_id', datatype, 'no_of_events']
df_signalcount['clone'] = df_signalcount['unique_id'].map(df_burst.groupby(['unique_id'])['clone'].first())
df_signalcount['track_length'] = df_signalcount['unique_id'].map(df_burst.groupby(['unique_id']).size())

df_signalcount.sort_values(by=[datatype], inplace=True)
df_signalcount_probability = df_signalcount.drop_duplicates(subset=['unique_id'], keep='last')
df_signalcount_probability.loc[df_signalcount_probability[datatype] == False, 'no_of_events'] = 0
df_signalcount_probability['norm_no_bursts'] = df_signalcount_probability['no_of_events'] / df_signalcount_probability[
    'track_length']
percentage_df = df_signalcount_probability.groupby('clone')['norm_no_bursts'].describe().reset_index()
percentage_df = percentage_df.merge(df_positions[['clone', 'distance', 'cp']], on='clone')
percentage_df.rename(columns={'mean': True}, inplace=True)
percentage_df.set_index('clone', inplace=True)

# scale trend-line to min and max value from my experiment (maybe fit instead?)
df_trendline_scaled = df_trendline.copy()
# Define the minimum and maximum values for scaling
min_value, min_cp = percentage_df.loc['6G3', [True, 'cp']].values
max_value, max_cp = percentage_df.loc['5G7', [True, 'cp']].values
# Extract the original y values corresponding to x1 and x2
min_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == round(min_cp,2), df_trendline_scaled.columns[1]].values[0]
max_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0.99, df_trendline_scaled.columns[1]].values[0]

# Calculate slope (m) and intercept (b)
m = (max_value - min_value) / (max_original - min_original)
b = min_value - m * min_original

# Apply the linear transformation to the entire y column
df_trendline_scaled['y_scaled'] = m * df_trendline_scaled.iloc[:, 1] + b

# probability of bursting
plt.figure(figsize=(1.57, 1))
plt.scatter(percentage_df['cp'], percentage_df[True], color='black', zorder=3, s=3)
plt.plot(df_trendline_scaled[0], df_trendline_scaled['y_scaled'], color='purple', linewidth=1, alpha=0.6, zorder=2)
plt.grid(alpha=0.2, zorder=1)
x_ticks = np.linspace(0, 1, 5)
plt.xticks(x_ticks)
plt.tick_params(axis='both', which='both', length=1.5)
plt.xlabel('Contact probability')
plt.ylabel('Burst probability')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, 'Fig1', 'probability_bursting_trendline.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()



# --- Kymograph of spotdetected per clone ---
df_clone = df_burst[(df_burst['clone'] == '5G7')]
df_clone = df_clone.groupby('unique_id').filter(lambda x: x.frame.count() >= 480)

kymograph = df_clone.pivot(index='unique_id', columns='frame', values='spotdetected_filtered_curated')
kymograph.dropna(axis=0, how='all', inplace=True)
kymograph = kymograph.fillna(5).astype(int)
kymograph.replace(5, np.nan, inplace=True)

#sort based on number of bursts
kymograph['count_ones'] = kymograph.sum(axis=1)
kymograph = kymograph.sort_values(by='count_ones', ascending=False)
kymograph = kymograph.drop(columns='count_ones')

plt.figure(figsize=(10, 6))
plt.imshow(kymograph, cmap='viridis', aspect='auto', interpolation='none')
x_ticks = np.arange(0, kymograph.shape[1], 100).astype(int)
plt.xticks(x_ticks, (x_ticks / 2).astype(int))

plt.tick_params(axis='both', labelsize=26)
plt.grid(False)
#plt.savefig(os.path.join(path_out, 'kymograph_5F11.png'))
plt.show()
plt.close()


# --- Burst size or amplitude---
# calculate stats for plotting
# drop the edge states of a trace: set first and last group of True\False values in each trace to nan and drop them
df_dropedge = df_burst.groupby(['unique_id'], group_keys=True).apply(
    lambda x: exclude_first_last(x, column=datatype)).reset_index(drop=True)
df_dropedge = df_dropedge.dropna(subset=datatype).reset_index(drop=True)

# calculate the duration of consecutive on/off states, includes integrated intensities and intensity maxima
df_signallength_dropedge = [
    signallength_includingintensities2(df_dropedge[df_dropedge['unique_id'] == i], column=datatype) for i
    in df_dropedge['unique_id'].unique()]
df_signallength_dropedge = pd.concat(df_signallength_dropedge, ignore_index=True)
df_signallength_dropedge['clone'] = df_signallength_dropedge['unique_id'].map(
    df_burst.groupby(['unique_id'])['clone'].first())
df_signallength_dropedge['bursting'] = df_signallength_dropedge['unique_id'].map(
    df_burst.groupby(['unique_id'])['bursting'].first())

# define what metric to use (sum (burst size) or max intensity(burst amplitude)
metric = 'corr_tracemax'  # corr_tracemax or corr_tracesum
measure = "size" if metric == 'corr_tracesum' else "amplitude"

# calculate confidence interval using bootstrapping
confidence_interval = []
for clone in clones_pretty:
    # calculate cdf and survival function
    cdf = df_signallength_dropedge[
        (df_signallength_dropedge[datatype] == True) & (df_signallength_dropedge['clone'] == clone)][
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
    confidence_interval.append(df)
confidence_interval = pd.concat(confidence_interval)

from scipy.stats import ks_2samp

# Calculate the observed KS statistic for each pair of clones
observed_ks_stats = []
observed_pvalues_stats = []

# Extract original data for KS test
original_data = df_signallength_dropedge[df_signallength_dropedge[datatype] == True]

for i, clone1 in enumerate(clones_pretty):
    for clone2 in clones_pretty[i+1:]:
        clone1_data = original_data[original_data['clone'] == clone1][metric]
        clone2_data = original_data[original_data['clone'] == clone2][metric]

        # Calculate the KS statistic
        ks_stat, pvalue = ks_2samp(clone1_data, clone2_data)
        observed_ks_stats.append([clone1, clone2, ks_stat])
        observed_pvalues_stats.append([clone1, clone2, pvalue])

print("Observed KS statistic:", observed_stat)
print("p-value:", p_value)

# plot
fig, ax = plt.subplots(figsize=(1.73, 1.5))
for id, clone in enumerate(clones_pretty):
    cdf = df_signallength_dropedge[
        (df_signallength_dropedge[datatype] == True) & (df_signallength_dropedge['clone'] == clone)][
        metric].value_counts(normalize=True).sort_index().cumsum()
    survival = 1 + cdf.iloc[0] - cdf
    plt.plot(survival, color=colors_gregory[id], linewidth=1)
    plt.fill_between(survival.index, confidence_interval[confidence_interval['clone'] == clone]['lower_bound'],
                     confidence_interval[confidence_interval['clone'] == clone]['upper_bound'], alpha=0.2,
                     color=colors_gregory[id])
# Set the y-axis to use plain decimal numbers if in log
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel(f'Burst {measure} (a.u.)')
plt.ylabel('Survival probability')
plt.tight_layout(pad=0.1)
#plt.savefig(os.path.join(path_out, f'burst_{measure}_linear.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()


# --- Burst frequency ---
df_signallength = df_burst.groupby(['unique_id']).apply(signallength_includingintensities2, column=datatype).reset_index(drop=True)
df_signallength['clone'] = df_signallength['unique_id'].map(df_burst.groupby(['unique_id'])['clone'].first())
df_signallength['bursting'] = df_signallength['unique_id'].map(df_burst.groupby(['unique_id'])['bursting'].first())

df_signalcount = df_signallength[['unique_id', datatype]].groupby('unique_id').value_counts().reset_index()
df_signalcount.columns = ['unique_id', datatype, 'no_of_events']
df_signalcount['clone'] = df_signalcount['unique_id'].map(df_burst.groupby(['unique_id'])['clone'].first())
df_signalcount['track_length'] = df_signalcount['unique_id'].map(df_burst.groupby(['unique_id']).size())

# calculate burst frequency
df_burstfrequency = df_signalcount.sort_values(by=datatype, ascending=False)
df_burstfrequency = df_burstfrequency.drop_duplicates(subset='unique_id')
# select rows where spotdetected curated is false and set no_of_events to 0
df_burstfrequency.loc[df_burstfrequency[df_burstfrequency[datatype] == False].index, 'no_of_events'] = 0
df_burstfrequency['burst_frequency_perh'] = df_burstfrequency['no_of_events'] / (df_burstfrequency['track_length']/120)

df_burstfrequency_avg = df_burstfrequency.groupby('clone')['burst_frequency_perh'].describe().reset_index()
df_burstfrequency_avg = df_burstfrequency_avg.merge(df_positions[['clone', 'distance', 'cp']], on='clone', how='left')

# scale trend-line to min and max value from my experiment (maybe fit instead?)
df_trendline_scaled = df_trendline.copy()
# Define the minimum and maximum values for scaling
min_value, min_cp = df_burstfrequency_avg.loc[df_burstfrequency_avg['clone'] == '6G3', ['mean', 'cp']].values[0]
max_value, max_cp = df_burstfrequency_avg.loc[df_burstfrequency_avg['clone'] == '5G7', ['mean', 'cp']].values[0]
# Extract the original y values corresponding to x1 and x2
min_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0, df_trendline_scaled.columns[1]].values[0]
max_original = df_trendline_scaled.loc[df_trendline_scaled.iloc[:, 0] == 0.99, df_trendline_scaled.columns[1]].values[0]

# Calculate slope (m) and intercept (b)
m = (max_value - min_value) / (max_original - min_original)
b = min_value - m * min_original

# Apply the linear transformation to the entire y column
df_trendline_scaled['y_scaled'] = m * df_trendline_scaled.iloc[:, 1] + b

plt.figure(figsize=(2, 1.5))
plt.scatter(df_burstfrequency_avg['cp'], df_burstfrequency_avg['mean'], color='black', s=3, zorder=2)
plt.plot(df_trendline_scaled[0], df_trendline_scaled['y_scaled'], color='purple', alpha=0.6, linewidth=1, zorder=1)
plt.grid(alpha=0.2, zorder=0)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Contact probability')
plt.ylabel('Burst frequency\n(counts/hour)')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, 'burstfreq_trendline.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# cv burst frequency
fig, ax = plt.subplots(figsize=(5, 5))
df_burstfrequency_avg['cv'] = df_burstfrequency_avg['std'] / df_burstfrequency_avg['mean']
for i, colors in enumerate(colors_gregory):
    plt.scatter(df_burstfrequency_avg.loc[i, 'cp'], df_burstfrequency_avg.loc[i, 'cv'], color=colors, s=200, linewidths=1,edgecolors='black', zorder=2)
plt.grid(alpha=0.2, zorder=1)
plt.xlabel('Contact Probability')
plt.ylabel('CV burst frequency')
plt.tight_layout()
plt.ylim(0.5, 11)
plt.xlim(-0.15, 1.15)
plt.savefig(os.path.join(path_out, 'cv_burstfreq.pdf'))
plt.show()
plt.close()


# --- Comparison Burst frequency, mean mRNA count and protein level ---
df_fish_avg = df_fish.groupby('clone')['spots_per_cell'].describe()
df_facs_exp_avg = df_facs_exp.groupby('Clone')['Mean'].describe()
df_burstfrequency_avg.set_index('clone', inplace=True)

# burst frequency and FISH data
common_index = df_fish_avg.index.intersection(df_burstfrequency_avg.index)
df_fish_avg_com = df_fish_avg.loc[common_index]
df_burstfrequency_avg_com = df_burstfrequency_avg.loc[common_index]
r2 = stats.pearsonr(df_fish_avg_com['mean'], df_burstfrequency_avg_com['mean'])[0] ** 2

fig, ax = plt.subplots()
#ax.errorbar(df_fish_avg_com['mean'], df_burstfrequency_avg_com['mean'], xerr=df_fish_avg_com['std'],
#            yerr=df_burstfrequency_avg_com['std'], fmt="o", color='black')
sns.regplot(df_fish_avg_com['mean'], df_burstfrequency_avg_com['mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}')
plt.xlabel('Mean number of mature mRNA')
plt.ylabel('Burst frequency (counts/hour)')
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig(os.path.join(path_out, 'corr_burstfreq_mRNA.pdf'))
plt.show()
plt.close()

# burst frequency and facs data
common_index = df_facs_exp_avg.index.intersection(df_burstfrequency_avg.index)
df_facs_exp_avg_com = df_facs_exp_avg.loc[common_index]
df_burstfrequency_avg_com = df_burstfrequency_avg.loc[common_index]
r2 = stats.pearsonr(df_facs_exp_avg_com['mean'], df_burstfrequency_avg_com['mean'])[0] ** 2

fig, ax = plt.subplots()
#ax.errorbar(df_facs_exp_avg_com['mean'], df_burstfrequency_avg_com['mean'], xerr=df_facs_exp_avg_com['std'],
#            yerr=df_burstfrequency_avg_com['std'], fmt="o", color='black')
sns.regplot(df_facs_exp_avg_com['mean'], df_burstfrequency_avg_com['mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}')
plt.xlabel('Mean protein levels (a.u.)')
plt.ylabel('Burst frequency (counts/hour)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'corr_burstfreq_protein.pdf'))
plt.show()
plt.close()

# FISH and FACS data
common_index = df_fish_avg.index.intersection(df_facs_exp_avg.index)
df_fish_avg_com = df_fish_avg.loc[common_index]
df_facs_exp_avg_com = df_facs_exp_avg.loc[common_index]
r2 = stats.pearsonr(df_fish_avg_com['mean'], df_facs_exp_avg_com['mean'])[0] ** 2

fig, ax = plt.subplots()
ax.errorbar(df_fish_avg_com['mean'], df_facs_exp_avg_com['mean'], xerr=df_fish_avg_com['std'],
            yerr=df_facs_exp_avg_com['std'], fmt="o", color='black')
sns.regplot(df_fish_avg_com['mean'], df_facs_exp_avg_com['mean'], ci=None, color='black',
            label=f'Rsquare={round(r2, 2)}')
plt.xlabel('Mean number of mature mRNA')
plt.ylabel('Mean protein levels (a.u.)')
plt.legend(loc='lower right')
plt.tight_layout()
#plt.savefig(os.path.join(path_out, 'corr_protein_mRNA.pdf'))
plt.show()
plt.close()

# statistics
df_signallength.groupby(['clone'])['unique_id'].nunique()
df_signallength.groupby(['clone', 'spotdetected_filtered_curated'])['spotdetected_filtered_curated'].value_counts()

def manipulate_string(s):
    parts = s.split('_')  # Split the string by '_'
    parts = parts[:-2]  # Remove the last three elements
    return '_'.join(parts)  # Join the remaining parts back together

# Apply the function to the column
df_burst['filename_replicates'] = df_burst['filename'].apply(manipulate_string)

df_burst.groupby(['clone'])['filename_replicates'].nunique()