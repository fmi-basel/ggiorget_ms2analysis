# Load all the packages
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = {'size': 6}
matplotlib.rc('font', **font)
mm = 1 / 25.4


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
    data.loc[data[column] == min_id, 'censored'] = 'leftcensored'
    data.loc[data[column] == max_id, 'censored'] = 'rightcensored'
    if min_id == max_id:
        data['censored'] = 'nonbursting'
    return data


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

#function that creates either 1,2 or 3 exponential fits based on given parameters
def oneexponential(t, T1):
    return np.exp(-t / T1)
def twoexponential(t, T1, T2, f):
    return f * np.exp(-t / T1) + (1 - f) * np.exp(-t / T2)

def threeexponential(t, T1, T2, T3, f1, f2, f3):
    return f1 * np.exp(-t / T1) + f2 * np.exp(-t / T2) + f3 * np.exp(-t / T3)


# Load data
path_out = '/Users/janatunnermann/Desktop/plots_paper/'

Fits = '1state'

# bursting data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data'
filename = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800_curated.csv'
df_burst = pd.read_csv(os.path.join(path, filename), dtype={'clone': 'str'}, low_memory=False)

# Fits from Gregory
path_fits = '/Volumes/ggiorget_scratch/Gregory/julia/bursting_manuscript/figures/initiationfits/'
df_1state_intensityfit = pd.read_csv(os.path.join(path_fits, '1state_ns_line_intensity_fit.csv'))
df_1state_intensityline = pd.read_csv(os.path.join(path_fits, '1state_ns_line_intensity_data.csv'))
df_1state_survivalon = pd.read_csv(os.path.join(path_fits, '1state_ns_line_survivalOn_fit.csv'))
df_1state_mRNA = pd.read_csv(os.path.join(path_fits, '1state_ns_meannascentRNA.csv'), index_col=False)

df_2state_intensityfit = pd.read_csv(os.path.join(path_fits, '2state_ns_line_intensity_fit.csv'))
df_2state_intensityline = pd.read_csv(os.path.join(path_fits, '2state_ns_line_intensity_data.csv'))
df_2state_survivalon = pd.read_csv(os.path.join(path_fits, '2state_ns_line_survivalOn_fit.csv'))
df_2state_mRNA = pd.read_csv(os.path.join(path_fits, '2tate_ns_meannascentRNA.csv'), index_col=False)

# which datatype to do the calculations with
datatype = 'spotdetected_filtered_curated'

df_burst = df_burst[df_burst['frame'] >= 120]
df_burst = df_burst[df_burst['frame'] <= 600]
df_burst = df_burst[df_burst[datatype].notna()]

# ---- Pre-processing ----
# posibility to resample tracks to lower frame rate or select only the long movies
# df = df.iloc[::6]
# df = df.groupby('unique_id').filter(lambda x: x.frame.count()==361).reset_index(drop=True)
# settings for colors
red_5g7 = (213 / 255, 87 / 255, 69 / 255)

# ---- KAPLAN-Meier on-rate ----
# sum-up length of states
df_signallength = [signallength(df_burst[df_burst['unique_id'] == i], column=datatype) for i in
                   df_burst['unique_id'].unique()]
df_signallength = pd.concat(df_signallength, ignore_index=True)
df_signallength['clone'] = df_signallength['unique_id'].map(df_burst.groupby(['unique_id'])['clone'].first())

# Indicate which values are incomplete/censored
df_signallength = [indicate_first_last(df_signallength[df_signallength['unique_id'] == i], column='signal_no') for i in
                   df_signallength['unique_id'].unique()]
df_signallength = pd.concat(df_signallength, ignore_index=True)

# decide on which data to use
# drop left censored
df_signallength = df_signallength[df_signallength.censored != 'leftcensored']
# keep right censored and non-censored
df_signallength.loc[df_signallength['censored'] == 'rightcensored', 'censored'] = 0
df_signallength.loc[df_signallength['censored'] == 'noncensored', 'censored'] = 1
# Either drop or keep non bursting
df_signallength = df_signallength[df_signallength.censored != 'nonbursting']

# - Fitting -
signal = True
measure = "on" if signal == True else "off"
# chose off times with at least one burst in trace
df_signallength_times = df_signallength[df_signallength[datatype] == signal]
df_signallength_times = df_signallength_times[df_signallength_times['clone'] == '5G7']

df_fit = df_1state_survivalon if Fits == '1state' else df_2state_survivalon

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(28 * mm, 1.5))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
km_ci_lower = kmf.confidence_interval_.iloc[:, 0].values
km_ci_lower[km_ci_lower == 0] = np.nan
km_ci_upper = kmf.confidence_interval_.iloc[:, 1].values
km_ci_upper[km_ci_upper == 0] = np.nan
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7, alpha=0.5)
ax.fill_between(km_times[1:], km_ci_lower[1:], km_ci_upper[1:], color=red_5g7, alpha=0.2)
ax.plot(df_fit['time(frame)'] / 2, df_fit['5G7'], color='black', linestyle='--')
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Time (min)')
plt.ylabel('Survival probability')
plt.title('Active time')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'{measure}-times_KM_{Fits}FIT.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ---- Intensity traces ----
data_line_int = df_1state_intensityline if Fits == '1state' else df_2state_intensityline
df_fit_int = df_1state_intensityfit if Fits == '1state' else df_2state_intensityfit

fig, ax = plt.subplots(figsize=(28 * mm, 1.5))
ax.plot(data_line_int['time(frame)'] / 2, data_line_int['5G7'], color=red_5g7)
ax.plot(df_fit_int['time(frame)'] / 2, df_fit_int['5G7'], color='black', linestyle='--')
plt.ylabel('Intensity (a.u.)')
plt.xlabel('Time (min)')
plt.title('Intensity increase')
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'Intensity-trace_overlay_{Fits}FIT.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# ---- Mean number mRNA per ts ----
df_fit_mRNA = df_1state_mRNA if Fits == '1state' else df_2state_mRNA
df_fit_mRNA = df_fit_mRNA.transpose()
df_fit_mRNA['color'] = [red_5g7] * len(df_fit_mRNA)
df_fit_mRNA.loc['model', 'color'] = 'darkgrey'

fig, ax = plt.subplots(figsize=(28 * mm, 1.5))
plt.bar(df_fit_mRNA.index, df_fit_mRNA[0], color=df_fit_mRNA['color'], zorder=2)
plt.ylabel('Mean # of RNA per TS')
plt.xlabel('')
plt.title('Nascent RNA')
plt.grid(alpha=0.2, zorder=1)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'mean_mRNA_{Fits}.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()



# OFF times - exponential
path_exponential = '/Volumes/ggiorget_scratch/Gregory/julia/bursting_manuscript/figures/exponentialfits/off_times/fit_summary'

df_1_exponential = pd.read_csv(os.path.join(path_exponential, 'km_1expfit.csv'))
df_1_exponential.replace(np.inf, 1000, inplace=True)
df_2_exponential = pd.read_csv(os.path.join(path_exponential, 'km_2expfit.csv'))
df_2_exponential.replace(np.inf, 1000, inplace=True)
df_3_exponential = pd.read_csv(os.path.join(path_exponential, 'km_3expfit.csv'))
df_3_exponential.replace(np.inf, 1000, inplace=True)


# - Fitting -
signal = False
measure = "on" if signal == True else "off"
# chose off times with at least one burst in trace
df_signallength_times = df_signallength[df_signallength[datatype] == signal]
df_signallength_times = df_signallength_times[df_signallength_times['clone'] == '5G7']


# one exp
result = df_1_exponential.loc[df_1_exponential['clone'] == '5G7']
T1 = result['time'].values[0]

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(35*mm, 35*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = oneexponential(km_times[1:], T1)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
ax.fill_between(km_times[1:], fit_line, y2=0.02, color='antiquewhite', alpha=0.8)
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Time (min)')
plt.ylabel('Survival probability')
plt.ylim(0.02,1)
plt.xlim(0)
plt.title('1-timescale')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'off-times_KM_1expFIT.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# zoom
fig, ax = plt.subplots(figsize=(20*mm, 17*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = oneexponential(km_times[1:], T1)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
plt.yscale('log')
plt.xlim(0, 10)
plt.ylim(0.5,1)
custom_ticks_y = [0.5, 0.75, 1.0]
custom_ticks_x = [0, 5, 10]
# Set custom ticks for y-axis
ax.set_xticks(custom_ticks_x)
ax.set_yticks(custom_ticks_y)
plt.minorticks_off()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)  # Disable scientific notation
formatter.set_useOffset(False)   # Disable offset
ax.yaxis.set_major_formatter(formatter)
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.01)
plt.savefig(os.path.join(path_out, f'off-times_KM_1expFIT_zoom.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# two exp
result = df_2_exponential.loc[df_2_exponential['clone'] == '5G7']
T1 = result['timeslow'].values[0]
T2 = result['timefast'].values[0]
f = result['pslow'].values[0]

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(35*mm, 35*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = twoexponential(km_times[1:], T1, T2, f)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
exp1 = f*oneexponential(km_times[1:], T1)
exp2 = (1-f)*oneexponential(km_times[1:], T2)
exp2_limit = exp2.copy()
exp2_limit[exp2_limit < 0.02] = 0.02
ax.fill_between(km_times[1:], exp2_limit, 0.02, color='antiquewhite', alpha=0.8)
exp21_limit = exp2+exp1.copy()
exp21_limit[exp21_limit < 0.02] = 0.02
ax.fill_between(km_times[1:], exp21_limit, exp2_limit,  color='lightblue', alpha=0.8)
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Time (min)')
plt.ylabel('Survival probability')
plt.ylim(0.02,1)
plt.xlim(0)
plt.title('2-timescales')
plt.legend()
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'off-times_KM_2expFIT.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# zoom
fig, ax = plt.subplots(figsize=(20*mm, 17*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = twoexponential(km_times[1:], T1, T2, f)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
plt.yscale('log')
plt.xlim(0, 10)
plt.ylim(0.5,1)
custom_ticks_y = [0.5, 0.75, 1.0]
custom_ticks_x = [0, 5, 10]
# Set custom ticks for y-axis
ax.set_xticks(custom_ticks_x)
ax.set_yticks(custom_ticks_y)
plt.minorticks_off()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)  # Disable scientific notation
formatter.set_useOffset(False)   # Disable offset
ax.yaxis.set_major_formatter(formatter)
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.01)
plt.savefig(os.path.join(path_out, f'off-times_KM_2expFIT_zoom.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()


# three exp
result = df_3_exponential.loc[df_3_exponential['clone'] == '5G7']
T1 = result['timeslow'].values[0]
T2 = result['timemiddle'].values[0]
T3 = result['timefast'].values[0]
f1 = result['pslow'].values[0]
f2 = result['pmiddle'].values[0]
f3 = result['pfast'].values[0]

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(figsize=(35*mm, 35*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = threeexponential(km_times[1:], T1, T2, T3, f1, f2, f3)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
exp1 = f1*oneexponential(km_times[1:], T1)
exp2 = f2*oneexponential(km_times[1:], T2)
exp3 = f3*oneexponential(km_times[1:], T3)
exp3_limit = exp3.copy()
exp3_limit[exp3_limit < 0.02] = 0.02
ax.fill_between(km_times[1:], exp3_limit, y2=.02, color='sandybrown', alpha=0.8)
exp32_limit = exp3+exp2.copy()
exp32_limit[exp32_limit < 0.02] = 0.02
ax.fill_between(km_times[1:], exp32_limit, exp3_limit, color='antiquewhite', alpha=0.8)
exp321_limit = exp3+exp2+exp1.copy()
exp321_limit[exp321_limit < 0.02] = 0.02
ax.fill_between(km_times[1:], exp321_limit, exp32_limit, color='lightblue', alpha=0.8)
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.xlabel('Time (min)')
plt.ylabel('Survival probability')
plt.ylim(0.02,1)
plt.xlim(0)
plt.title('3-timescales')
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(path_out, f'off-times_KM_3expFIT.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# zoom
fig, ax = plt.subplots(figsize=(20*mm, 17*mm))
kmf.fit((df_signallength_times['framecount'] / 2), event_observed=df_signallength_times['censored'])
km_times = kmf.survival_function_.index.values
km_survival_probs = kmf.survival_function_['KM_estimate'].values
km_survival_probs[km_survival_probs == 0] = np.nan
fit_line = threeexponential(km_times[1:], T1, T2, T3, f1, f2, f3)
ax.plot(km_times[1:], km_survival_probs[1:], color=red_5g7)
ax.plot(km_times[1:], fit_line, color='black', linestyle='--')
plt.yscale('log')
plt.xlim(0, 10)
plt.ylim(0.5,1)
custom_ticks_y = [0.5, 0.75, 1.0]
custom_ticks_x = [0, 5, 10]
# Set custom ticks for y-axis
ax.set_xticks(custom_ticks_x)
ax.set_yticks(custom_ticks_y)
plt.minorticks_off()
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)  # Disable scientific notation
formatter.set_useOffset(False)   # Disable offset
ax.yaxis.set_major_formatter(formatter)
plt.grid(alpha=0.2)
plt.tick_params(axis='both', which='both', length=1.5)  # Adjust tick length
plt.tight_layout(pad=0.01)
plt.savefig(os.path.join(path_out, f'off-times_KM_3expFIT_zoom.pdf'), bbox_inches='tight', transparent=True)
plt.show()
plt.close()


