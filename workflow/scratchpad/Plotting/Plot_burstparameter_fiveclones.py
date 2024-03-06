"""
This script plots data regarding burst parameter and other exploratory things (track-length, intensity distribution,
on-/off-times, signal intensity.
Some summary statistics are calculated, from data where the edge cases are excluded (I don't know for how long these
states are present). These statistics include average signal duration, mean GFP levels, average track length. Lastly,
plots are generated regarding track lengths, on-/off time durations, burst probability plots and burst signal intensity
(integrated and amplitude intensity).
"""

# Load all the packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils_postprocessing import exclude_first_last, signallength_includingintensities2

# Load data
path_out = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/plotting_figures_movies/burst/fiveclones'
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/summarized_tracks/fiveclones_old'
# path = '/Users/janatunnermann/Desktop/'
filename = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_combined_fiveclones_postprocessed.csv'
df = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'})


selected_tracks = df[(df['frame']>120) & (df['frame']<=600)].reset_index(drop=True)
selected_tracks['duration'] = selected_tracks.groupby('unique_id')['frame'].transform(lambda x: x.max() - x.min() + 1)
long_tracks = selected_tracks[selected_tracks['duration'] >= 480]

# Count the number of such tracks
num_long_tracks = long_tracks[['clone', 'unique_id']].drop_duplicates().groupby('clone').size()



# which datatype to do the calculations with
# datatype = ['spotdetected', 'spotdetected_filtered', 'spotdetected_filtered_curated]
datatype = 'spotdetected_filtered'

bursting = df.groupby(['unique_id']).apply(lambda x: x[datatype].any()).reset_index()
bursting.columns = ['unique_id', 'bursting']
df = df.merge(bursting, on='unique_id')

clones = np.unique(df['clone'])
clones = np.roll(clones, 1)

# ---- Pre-processing ----
# possibility to resample tracks to lower frame rate or select only the long movies
# df = df.iloc[::6]
# df = df.groupby('unique_id').filter(lambda x: x.frame.count() <= 601)

# create a df summarizing cells statistics in regard to GFP levels
df_gfp = df[df['frame']==0][
    ['filename', 'unique_id', 'clone', 'mean_gfp', 'sd_gfp', 'median_gfp', 'mad_gfp', 'integrated_intensity_gfp', 'bursting',
     'area_completecell']]
df_gfp = df_gfp.merge(df.groupby(['unique_id']).size().reset_index())
df_gfp.columns = [*df_gfp.columns[:-1], 'track_length']
# possible to drop cells from a certain day
# df_gfp = df_gfp[~df_gfp.filename.str.startswith('20221027')]
# df_gfp.to_csv(os.path.join(path_out, 'summary_gfp.csv'), index=False)


# ---- Summarizing statistics ----
# drop the edge states of a trace: set first and last group of True\False values in each trace to nan and drop them
df_dropedge = df.groupby(['unique_id'], group_keys=True).apply(
    lambda x: exclude_first_last(x, column=datatype)).reset_index(drop=True)
df_dropedge = df_dropedge.dropna(subset=datatype).reset_index(drop=True)


# calculate the duration of consecutive on/off states, includes integrated intensities
df_signallength_dropedge = [
    signallength_includingintensities2(df_dropedge[df_dropedge['unique_id'] == i], column=datatype) for i
    in df_dropedge['unique_id'].unique()]
df_signallength_dropedge = pd.concat(df_signallength_dropedge, ignore_index=True)
df_signallength_dropedge['clone'] = df_signallength_dropedge['unique_id'].map(
    df.groupby(['unique_id'])['clone'].first())
df_signallength_dropedge['bursting'] = df_signallength_dropedge['unique_id'].map(
    df.groupby(['unique_id'])['bursting'].first())

df_signallength = [signallength_includingintensities2(df[df['unique_id'] == i], column=datatype) for i in
                   df['unique_id'].unique()]
df_signallength = pd.concat(df_signallength, ignore_index=True)
df_signallength['clone'] = df_signallength['unique_id'].map(df.groupby(['unique_id'])['clone'].first())
df_signallength['bursting'] = df_signallength['unique_id'].map(df.groupby(['unique_id'])['bursting'].first())

Check = df_signallength[df_signallength[datatype]==True]
Check['filename'] = Check['unique_id'].map(df.groupby(['unique_id'])['filename'].first())
Check['track_id'] = Check['unique_id'].map(df.groupby(['unique_id'])['track_id'].first())
Check = Check[['filename', 'track_id', 'framecount']]
Check['date'] = Check['filename'].str.rsplit(pat='_', n=-1, expand=True)[0]

Check2 = Check[(Check['framecount']>14) & (Check['framecount']<20)]

# calculate the nuber of on/off states in each trace
df_signalcount_dropedge = df_signallength_dropedge[['unique_id', datatype]].groupby(
    'unique_id').value_counts().reset_index()
df_signalcount_dropedge.columns = ['unique_id', datatype, 'no_of_events']
df_signalcount_dropedge['clone'] = df_signalcount_dropedge['unique_id'].map(df.groupby(['unique_id'])['clone'].first())
df_signalcount_dropedge['track_length'] = df_signalcount_dropedge['unique_id'].map(df.groupby(['unique_id']).size())

df_signalcount = df_signallength[['unique_id', datatype]].groupby('unique_id').value_counts().reset_index()
df_signalcount.columns = ['unique_id', datatype, 'no_of_events']
df_signalcount['clone'] = df_signalcount['unique_id'].map(df.groupby(['unique_id'])['clone'].first())
df_signalcount['track_length'] = df_signalcount['unique_id'].map(df.groupby(['unique_id']).size())

df_signallength[['clone', 'framecount']].groupby('clone').value_counts()
# some summary statistics
# summary signal duration per clone
signal_duration_clone_dropedge = df_signallength_dropedge[['clone', datatype, 'framecount']].groupby(
    ['clone', datatype]).describe().reset_index()
signal_duration_clone = df_signallength[['clone', datatype, 'framecount']].groupby(
    ['clone', datatype]).describe().reset_index()
# Mean GFP level per clone
gfp_level = df_gfp.drop('unique_id', axis=1).groupby('clone').describe().reset_index()
df_signalcount[(df_signalcount[datatype] == False) & (df_signalcount['track_length'] == 601)].groupby('clone')[
    'no_of_events'].describe().reset_index()

# number of traces
len(np.unique(df['unique_id']))

# How many traces show bursting
df.groupby(['unique_id']).first().groupby('clone')['bursting'].value_counts()

# ---- Plotting ----
# -- Track-length --
# Track length distribution
plt.hist(df.groupby(['unique_id']).size(), bins=60)
plt.xlabel('Track length (frames)')
plt.ylabel('count')
plt.title('Track length distribution')
plt.show()
# plt.savefig(os.path.join(path_out, 'track_length_distribution.pdf'))
plt.close()

for clone in clones:
    plt.hist(df[df['clone'] == clone].groupby(['unique_id']).size(), bins=60, label=f'{clone}', alpha=0.5)
plt.xlabel('Track length (frames)')
plt.ylabel('count')
plt.legend()
plt.title('Track length distribution')
plt.show()
# plt.savefig(os.path.join(path_out, 'track_length_distribution_clone.pdf'))
plt.close()

# effective track length distribution after excluding edges
plt.hist(df_dropedge.groupby(['unique_id']).size(), bins=60)
plt.xlabel('Track length (frames)')
plt.ylabel('count')
plt.title('Effective track length distribution')
plt.show()
# plt.savefig(os.path.join(path_out, 'track_length_distribution_effective.pdf'))
plt.close()

for clone in clones:
    plt.hist(df_dropedge[df_dropedge['clone'] == clone].groupby(['unique_id']).size(), bins=60,
             label=f'{clone}', alpha=0.5)
plt.xlabel('Track length (min)')
plt.ylabel('count')
plt.legend()
plt.title('Track length distribution')
plt.show()
# plt.savefig(os.path.join(path_out, 'track_length_distribution_effective_clone.pdf'))
plt.close()

# Intensity distribution
plt.hist(df[df['spotdetected_filtered'] == True]['corr_trace'],
         bins=range(int(min(df['corr_trace'])), int(max(df['corr_trace'])) + 10, 30),
         label='Spot called', alpha=0.5, log=True)
plt.hist(df[df['spotdetected_filtered'] == False]['corr_trace'],
         bins=range(int(min(df['corr_trace'])), int(max(df['corr_trace'])) + 10, 30),
         label='Background called', alpha=0.5, log=True)
plt.xlabel('Intensity (a.u.)')
plt.ylabel('log count')
plt.legend()
plt.title('Intensity distribution')
plt.show()
# plt.savefig(os.path.join(path_out, 'Intensity_distribution_spotbackground.pdf'))
plt.close()

# -- Percentage cells bursting --
percentage_df = df.groupby(['unique_id']).first().groupby('clone')['bursting'].value_counts().unstack(fill_value=0)
percentage_df = percentage_df.div(percentage_df.sum(axis=1), axis=0) * 100

percentage_df.plot(kind='bar', stacked=True)
plt.xlabel('')
plt.ylabel('Percentage')
plt.title('Percentage of cells that burst')
plt.legend(title='Bursting', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path_out, 'percentage_cellburst.pdf'))
plt.close()

# -- On-signal length --
# dataframe = df_signallength
dataframe = df_signallength_dropedge
bins = np.arange(0, 601, 1)

# histogram
plt.figsize = (10, 10)
for clone in clones:
    plt.hist(dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)]['framecount'],
             bins=bins, label=f'{clone}', alpha=0.5, density=True)
plt.xlabel('On-time (frames)')
plt.ylabel('Density count')
plt.title('On-time distribution')
plt.legend()
plt.rcParams.update({'font.size': 12})
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'on_hist.pdf'))
plt.show()
plt.close()

# log-log histogram
for clone in clones:
    plt.hist(dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)]['framecount'],
             bins=bins, label=f'{clone}', alpha=0.5, density=True, log=True)
plt.xlabel('On-time (log frames)')
plt.ylabel('Log density count')
plt.title('On-time distribution')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.xscale('log')
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'on_hist_loglog.pdf'))
plt.show()
plt.close()

# CDF plot
for clone in clones:
    plt.plot(
        (dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)][
             'framecount'] / 2).value_counts(normalize=True).sort_index().cumsum(),
        label=f'{clone}')
plt.xlabel('On-time (min)')
plt.ylabel('Cumulative probability')
plt.title('On-time distribution')
plt.rcParams.update({'font.size': 12})
plt.legend(loc='lower right')
plt.xlim(-1, 10)
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'on_cdf_zoom.pdf'))
plt.show()
plt.close()

# -- Off-signal length --
# histogram
plt.figsize = (10, 10)
for clone in clones:
    plt.hist(dataframe[(dataframe[datatype] == False) & (dataframe['clone'] == clone)]['framecount'],
             bins=bins, label=f'{clone}', alpha=0.5, density=True)
plt.xlabel('Off-time (frames)')
plt.ylabel('Density count')
plt.title('Off-time distribution')
plt.rcParams.update({'font.size': 12})
plt.tight_layout()
plt.legend()
plt.show()
# plt.savefig(os.path.join(path_out, 'off_hist.pdf'))
plt.close()

# log-log histogram
plt.figsize = (10, 10)
for clone in clones:
    plt.hist(
        np.log(dataframe[(dataframe[datatype] == False) & (dataframe['clone'] == clone)]['framecount']),
        bins=bins, label=f'{clone}', alpha=0.5, density=True, log=True)
plt.xlabel('Off-time (log frames)')
plt.ylabel('Log density count')
plt.title('Off-time distribution')
plt.rcParams.update({'font.size': 12})
plt.tight_layout()
plt.legend()
plt.xscale('log')
# plt.savefig(os.path.join(path_out, 'off_hist_loglog.pdf'))
plt.show()
plt.close()

# CDF plot
for clone in clones:
    plt.plot(
        (dataframe[(dataframe[datatype] == False) & (dataframe['clone'] == clone)][
             'framecount'] / 2).value_counts(normalize=True).sort_index().cumsum(),
        label=f'{clone}')
plt.xlabel('Off-time (min)')
plt.ylabel('Cumulative probability')
plt.title('Off-time distribution')
plt.rcParams.update({'font.size': 12})
plt.legend()
# plt.ylim(0.35, 1.05)
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'off_cdf.pdf'))
plt.show()
plt.close()

# burst probability plots
# use all tracks and normalize for track length
df_signalcount.sort_values(by=[datatype], inplace=True)
df_signalcount_probability = df_signalcount.drop_duplicates(subset=['unique_id'], keep='last')
df_signalcount_probability.loc[df_signalcount_probability[datatype] == False, 'no_of_events'] = 0
df_signalcount_probability['norm_no_bursts'] = df_signalcount_probability['no_of_events'] / df_signalcount_probability[
    'track_length']

for clone in clones:
    plt.scatter(
        (df_signalcount_probability[df_signalcount_probability['clone'] == clone]['norm_no_bursts']).value_counts(
            normalize=True).sort_index().index,
        (df_signalcount_probability[df_signalcount_probability['clone'] == clone]['norm_no_bursts']).value_counts(
            normalize=True).sort_index().values,
        label=f'{clone}')
plt.xlabel('Normalized number of bursts (no./track length)')
plt.ylabel('Probability')
plt.title('No. of on-times')
plt.rcParams.update({'font.size': 12})
plt.legend()
# plt.ylim(0, 0.1)
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Numberof_ontimes_normalizedtracklength_zoom.pdf'))
plt.show()
plt.close()

for clone in clones:
    plt.scatter(
        (df_signalcount_probability[df_signalcount_probability['clone'] == clone]['norm_no_bursts']).value_counts(
            normalize=True).sort_index(ascending=False).index,
        (df_signalcount_probability[df_signalcount_probability['clone'] == clone]['norm_no_bursts']).value_counts(
            normalize=True).sort_index(ascending=False).values.cumsum(),
        label=f'{clone}')
plt.xlabel('Normalized number of bursts (no./track length)')
plt.ylabel('Cumulative probability')
plt.title('No. of on-times')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Numberof_ontimes_normalizedtracklength_cdf.pdf'))
plt.show()
plt.close()

for clone in clones:
    plt.scatter(
        (df_signalcount_probability[
            (df_signalcount_probability['clone'] == clone) & (df_signalcount_probability['track_length'] > 595)][
            'no_of_events']).value_counts(
            normalize=True).sort_index(ascending=False).index,
        (df_signalcount_probability[
            (df_signalcount_probability['clone'] == clone) & (df_signalcount_probability['track_length'] > 595)][
            'no_of_events']).value_counts(
            normalize=True).sort_index(ascending=False).values.cumsum(),
        label=f'{clone}')
plt.xlabel('Number of bursts')
plt.ylabel('Cumulative probability')
plt.title('No. of burst in 5h tracks')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Numberof_ontimes_4-9_htracks_cdf.pdf'))
plt.show()
plt.close()

"""
# use tracks that are 1h long only
plt.scatter(
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '6A12')][
        'no_of_events']).value_counts(normalize=True).sort_index().index,
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '6A12')][
        'no_of_events']).value_counts(normalize=True).sort_index().values,
    label='7.5 kb upstream')
plt.scatter(
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '5F11')][
        'no_of_events']).value_counts(normalize=True).sort_index().index,
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '5F11')][
        'no_of_events']).value_counts(normalize=True).sort_index().values,
    label='150 kb upstream')
plt.xlabel('Number of bursts')
plt.ylabel('Probability')
plt.title('Off-time distribution')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Numberof_ontimes_normalizedtracklength_1h.pdf'))
plt.show()
plt.close()

plt.plot(
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '6A12')][
        'no_of_events']).value_counts(normalize=True).sort_index(ascending=False).index,
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '6A12')][
        'no_of_events']).value_counts(normalize=True).sort_index(ascending=False).values.cumsum(),
    label='7.5 kb upstream')
plt.plot(
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '5F11')][
        'no_of_events']).value_counts(normalize=True).sort_index(ascending=False).index,
    (df_signalcount_probability[
        (df_signalcount_probability['track_length'] == 361) & (df_signalcount_probability['clone'] == '5F11')][
        'no_of_events']).value_counts(normalize=True).sort_index(ascending=False).values.cumsum(),
    label='150 kb upstream')
plt.xlabel('Normalized number of bursts (no./track length)')
plt.ylabel('Cumulative probability')
plt.title('No. of on-times')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Numberof_ontimes_normalizedtracklength_1h_cdf.pdf'))
plt.show()
plt.close()
"""

# -- Burst intensity --
# histogram of integrated burst intensity
for clone in clones:
    plt.hist(
        dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)]['corr_tracesum'],
        label=f'{clone}', alpha=0.5, bins=np.arange(0, 100000, 500), density=True)
plt.xlabel('Integrated intensity of burst (a.u.)')
plt.ylabel('Density count')
plt.title('Integrated burst intensity distribution')
plt.legend()
plt.xlim(-1000, 20000)
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'burstsize_integratedint_hist.pdf'))
plt.show()
plt.close()

# CDF plot of integrated burst intensity
for clone in clones:
    plt.plot(
        (dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)][
             'corr_tracesum'].value_counts(normalize=True).sort_index().cumsum()), label=f'{clone}')
plt.xlabel('Integrated intensity (a.u.)')
plt.ylabel('Cumulative probability')
plt.title('Integrated intensity distribution')
plt.rcParams.update({'font.size': 12})
plt.legend()
# plt.xlim(-1000, 20000)
plt.tight_layout()
#plt.savefig(os.path.join(path_out, 'burstsize_integratedint_cdf.pdf'))
plt.show()
plt.close()

#Pretty Plot
clones_pretty = clones[1:5]
green_5e10 = (111 / 255, 188 / 255, 133 / 255)
blue_5f11 = (113 / 255, 171 / 255, 221 / 255)
purple_5g3 = (156 / 255, 107 / 255, 184 / 255)
red_5g7 = (213 / 255, 87 / 255, 69 / 255)
colors_gregory = [green_5e10, blue_5f11, purple_5g3, red_5g7]


confidence_interval = []
for clone in clones_pretty:
    cdf = dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)]['mean_corrtracesum'].value_counts(
     normalize=True).sort_index().cumsum()
    cdf.reset_index()
    survival = 1+cdf.iloc[0]-cdf

    # Number of bootstrap samples
    num_samples = 10000  # You can adjust this as needed

    #Create an array to store the bootstrapped CDF values
    #bootstrap_cdfs = np.zeros((num_samples, len(cdf)))
    bootstrap_survivals = np.zeros((num_samples, len(survival)))
    # Perform bootstrapping
    for i in range(num_samples):
        # Resample with replacement from your original data
        bootstrap_sample = np.random.choice(survival.index, size=len(survival), replace=True)

        # Calculate the CDF of the bootstrap sample
        bootstrap_survival = pd.Series(bootstrap_sample).value_counts(normalize=True).sort_index().cumsum()
        bootstrap_survival = 1+bootstrap_survival.iloc[0]-bootstrap_survival

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

import matplotlib.ticker as ticker
font_properties = plt.rcParams['font.family']

fig, ax = plt.subplots(figsize=(6, 5))
plt.rcParams.update({'font.size': 16})
for id, clone in enumerate(clones_pretty):
    cdf = dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)][
             'mean_corrtracesum'].value_counts(normalize=True).sort_index().cumsum()
    survival = 1+cdf.iloc[0]-cdf
    plt.plot(survival, color=colors_gregory[id], linewidth=3)
    plt.fill_between(survival.index, confidence_interval[confidence_interval['clone']==clone]['lower_bound'], confidence_interval[confidence_interval['clone']==clone]['upper_bound'], alpha=0.2, color=colors_gregory[id])
plt.yscale('log')
y_formatter = ticker.ScalarFormatter(useMathText=False)
y_formatter.set_scientific(False)
y_formatter.set_powerlimits((-2, 2))  # Adjust this limit as needed
ax.yaxis.set_major_formatter(y_formatter)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'burstsize_integratedint_log_annualmeeting.pdf'))
#plt.show()
plt.close()

# histogram of max burst intensity (amplitude)
for clone in clones:
    plt.hist(
        dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)]['corr_tracemax'],
        label=f'{clone}', alpha=0.5, bins=80, density=True)
plt.xlabel('Intensity amplitude (a.u.)')
plt.ylabel('Density count')
plt.title('Intensity amplitude of a burst - distribution')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'burstsize_amplitudeint_hist.pdf'))
plt.show()
plt.close()

# CDF plot of max burst intensity (amplitude)
for clone in clones:
    plt.plot(
        (dataframe[(dataframe[datatype] == True) & (dataframe['clone'] == clone)][
             'corr_tracemax'].value_counts(normalize=True).sort_index().cumsum()), label=f'{clone}')
plt.xlabel('Intensity amplitude (a.u.)')
plt.ylabel('Cumulative probability')
plt.title('Intensity amplitude distribution')
plt.rcParams.update({'font.size': 12})
plt.legend()
plt.tight_layout()
#plt.savefig(os.path.join(path_out, 'burstsize_amplitudeint_cdf.pdf'))
plt.show()
plt.close()

# -- GFP intensity --
# plot GFP intensity and bursting
sns.swarmplot(data=df_gfp, x='clone', y='mean_gfp')
# plt.xticks(ticks=[0, 1], labels=['-150 kb', '-7.5 kb'])
plt.xlabel('')
plt.ylabel('Mean GFP intensity (a.u.)')
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'GFP_meanint_clone.pdf'))
plt.show()
plt.close()

sns.swarmplot(data=df_gfp, x='clone', y='integrated_intensity_gfp')
plt.xlabel('')
plt.ylabel('Total integrated GFP intensity (a.u.)')
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'GFP_integratedint_clone.pdf'))
plt.show()
plt.close()

sns.swarmplot(data=df_gfp, x='clone', y='mean_gfp', hue='bursting', dodge=True, size=2)
plt.xlabel('')
plt.ylabel('Mean GFP intensity (a.u.)')
plt.tight_layout()
#plt.savefig(os.path.join(path_out, 'GFP_meanint_clone_bursting.pdf'))
plt.show()
plt.close()

plt.scatter(df_gfp['integrated_intensity_gfp'], df_gfp['area_completecell'])
plt.xlabel('Total integrated GFP intensity (a.u.)')
plt.ylabel('Cell area (pixel)')
plt.tight_layout()
# plt.savefig(os.path.join(path_out, 'Mean_GFPint_clone.pdf'))
plt.show()
plt.close()

# -- Cell background intensity (bleaching) --
# plot the average mean_completecell intensity over frames
for clone in clones:
    plt.plot(df[df['clone'] == clone].groupby(['frame'])['mean_completecell'].mean(), label=f'{clone}')
plt.xlabel('Frame')
plt.ylabel('Mean cell intensity (a.u.)')
plt.title('Mean cell intensity over time')
plt.legend()
plt.tight_layout()
# plt.xlim(0, 600)
# plt.savefig(os.path.join(path_out, 'Mean_cellint_clone_time.pdf'))
plt.show()
plt.close()

plt.plot(df.groupby(['frame'])['mean_completecell'].mean())
plt.xlabel('Frame')
plt.ylabel('Mean cell intensity (a.u.)')
plt.title('Mean cell intensity over time')
plt.tight_layout()
# plt.xlim(0, 600)
# plt.savefig(os.path.join(path_out, 'Mean_cellint_time.pdf'))
plt.show()
plt.close()
