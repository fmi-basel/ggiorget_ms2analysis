# This script compares sub-sampled data to the original. Every nth data point is left out and linearly interpolated to
# model longer frame rates. Then, the sub-sampled data is compared to the original track by calculating the RMSE (root-
# mean-square-error). I also calculate the RMSE between two sets of 20s subsampled data.

# Let's start with loading all the packages and define the root-mean-square-error
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def root_mean_squared_error(act, pred):
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = np.nanmean(differences_squared)
    rmse_val = np.sqrt(mean_diff)
    return rmse_val


# The original tracks have a frame rate of 10s. Hence, only frame rates divisible by 10 can be modeled. Define here
# frame rates to be modeled. (20-1000s, 10s steps)
frame_rates = np.arange(20, 300, 10)

# Load combined data of 4 movies, rename columns for readability later on and calculate a unique cell ID across movies
path = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/Processed/20221027/Flatfieldcorrected/CorrectionForReadout/output_combined_sub/'
filename = '20221027_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF646i_6A12_combined-corrcom-sub.csv'
df_tracks = pd.read_csv(os.path.join(path, filename))
df_tracks = df_tracks.rename(columns={'Median_spot': '10s_spot', 'Median_background': '10s_background'})
df_tracks['ID'] = df_tracks['Cell_ID_unique'].astype(str) + df_tracks['filename']

### Plot different frame rates on example traces ###
movie = '20221027_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF646i_6A12_1_FullseqTIRF-Cy5-mCherryGFPWithSMB_s2_MAX.tiff'
cell_ID = 4
df_single_cell = df_tracks[(df_tracks['filename'] == movie) & (df_tracks['Cell_ID_labelimage'] == cell_ID)].reset_index(
    drop=True)

plt.plot(df_single_cell['frame'], df_single_cell['10s_spot'], label='10 s')
plt.plot(df_single_cell['frame'][0::3], df_single_cell['10s_spot'][0::3] + 1000, label='30 s')
#plt.plot(df_single_cell['frame'][0::4], df_single_cell['10s_spot'][0::4] + 2000, label='40 s')
#plt.plot(df_single_cell['frame'][0::5], df_single_cell['10s_spot'][0::5] + 2000, label='50 s')
plt.plot(df_single_cell['frame'][0::7], df_single_cell['10s_spot'][0::7] + 2000, label='70 s')
plt.legend()
plt.ylabel('Intensity of spot roi (a.u.)')
plt.xlabel('frame')
#plt.show()
plt.savefig(os.path.join(path, 'plots', 'Framerate_Example-trace2.pdf'))
plt.close()

### sub-sample data with defined frame rates ###
# sub-sampling is done cell-by-cell to ensure linear interpolation of missing frames isn't done between tracks, also
# background and spot traces are treated independently
df_resampled_rates = pd.DataFrame()
for cell in np.unique(df_tracks['ID']):
    df_resampled = df_tracks[df_tracks['ID'] == cell][['ID', 'frame', '10s_spot', '10s_background']]
    for rate in frame_rates:
        df_resampled[str(rate) + 's_spot'] = df_tracks['10s_spot'].iloc[::int(rate / 10)]
        df_resampled[str(rate) + 's_background'] = df_tracks['10s_background'].iloc[::int(rate / 10)]
    df_resampled = df_resampled.interpolate()
    df_resampled_rates = pd.concat([df_resampled_rates, df_resampled], ignore_index=True)

# cell by cell calculate RMSE comparing original 10s frame rate to given frame rates, there will be a warning for NaNs.
# Those come from the sub-sampling and can be ignored
df_cell_RMSE = pd.DataFrame()
for cell in np.unique(df_tracks['ID']):
    df_cell = df_resampled_rates[df_resampled_rates['ID'] == cell]
    for step, rate in enumerate(frame_rates):
        RMSE_spot = root_mean_squared_error(df_cell['10s_spot'],
                                            df_cell[str(rate) + 's_spot'])
        RMSE_background = root_mean_squared_error(df_cell['10s_background'],
                                                  df_cell[str(rate) + 's_background'])
        df_cell_RMSE = pd.concat([df_cell_RMSE, pd.DataFrame(
            {'cell': [cell], 'frame rate': [rate], 'RMSE spot': [RMSE_spot], 'RMSE background': [RMSE_background]})],
                                 ignore_index=True)
# So far, I differed between spot and background, now combine those two for calculation of the mean of RMSE
df_cell_RMSE_combined = pd.melt(df_cell_RMSE, id_vars=['cell', 'frame rate'],
                                value_vars=['RMSE spot', 'RMSE background'], value_name='RMSE combined')
# calculate the mean of RMSE, either for spot, background or the combination
df_RMSE = df_cell_RMSE.groupby('frame rate').mean().reset_index()
df_RMSE['RMSE combined'] = df_cell_RMSE_combined.groupby('frame rate').mean().reset_index()['RMSE combined']

# Plot RMSE of different frame rates for spot, background or combination of both
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE spot'], label='spot')
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE background'], label='background')
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE combined'], label='combined')
plt.legend()
plt.ylabel('RMSE between frame rate and 10s frame rate')
plt.xlabel('frame rate')
plt.show()
# plt.savefig(os.path.join(path, 'plots', 'RMSE_framerate.pdf'))
plt.close()


### Comparing 20s subsampled data with itself ###
# Since I don't know how to set the above calculated RMSE into context, I want to compare the 20s frame rate with itself
# to get a feeling for high frequency noise

# resample data
df_resampled_20s = pd.DataFrame()
for cell in np.unique(df_tracks['ID']):
    df_cell = df_tracks[df_tracks['ID'] == cell][['ID', 'frame', '10s_spot', '10s_background']]
    for rate in frame_rates:
        df_cell['spot_1'] = df_cell['10s_spot'].iloc[::2]
        df_cell['spot_2'] = df_cell['10s_spot'].iloc[1::2]
        df_cell['background_1'] = df_cell['10s_background'].iloc[::2]
        df_cell['background_2'] = df_cell['10s_background'].iloc[1::2]
    df_cell = df_cell.interpolate()
    df_resampled_20s = pd.concat([df_resampled_20s, df_cell])

# calculate RMSE per cell
df_cell_20s_RMSE = pd.DataFrame()
for cell in np.unique(df_resampled_20s['ID']):
    df_cell = df_resampled_20s[df_resampled_20s['ID'] == cell]
    RMSE_spot = root_mean_squared_error(df_cell['spot_1'], df_cell['spot_2'])
    RMSE_background = root_mean_squared_error(df_cell['background_1'], df_cell['background_2'])
    df_cell_20s_RMSE = pd.concat([df_cell_20s_RMSE, pd.DataFrame(
        {'cell': [cell], 'RMSE spot': [RMSE_spot], 'RMSE background': [RMSE_background]})], ignore_index=True)

mean_spot = np.mean(df_cell_20s_RMSE['RMSE spot'])
mean_background = np.mean(df_cell_20s_RMSE['RMSE background'])
mean_combined = np.mean(pd.concat([df_cell_20s_RMSE['RMSE spot'], df_cell_20s_RMSE['RMSE background']]))

# Plot RMSE of different frame rates
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE spot'], label='spot')
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE background'], label='background')
plt.plot(df_RMSE['frame rate'], df_RMSE['RMSE combined'], label='combined')
plt.hlines(y=mean_spot, xmin=20, xmax=300, colors='blue', linestyles='--')
plt.hlines(y=mean_background, xmin=20, xmax=300, colors='orange', linestyles='--')
plt.hlines(y=mean_combined, xmin=20, xmax=300, colors='green', linestyles='--')
# plt.vlines(x=70, ymin=20, ymax=50, colors='r', linestyles='--')
plt.legend()
plt.ylabel('RMSE between frame rate and 10s frame rate')
plt.xlabel('frame rate')
plt.show()
# plt.savefig(os.path.join(path, 'plots', 'RMSE_framerate_10sec-noise.pdf'))
plt.close()


# Plot example traces of 20s sub-sampled data (even and uneven frames)
df_single_cell_20s = df_resampled_20s[(df_tracks['filename'] == movie) & (df_tracks['Cell_ID_labelimage'] == cell_ID)].reset_index(
    drop=True)
plt.plot(df_single_cell_20s['frame'], df_single_cell_20s['spot_1'], label='uneven frames')
plt.plot(df_single_cell_20s['frame'], df_single_cell_20s['spot_2'], label='even frames')
plt.legend()
plt.ylabel('Intensity (a.u.)')
plt.xlabel('frame')
#plt.show()
plt.savefig(os.path.join(path, 'plots', '20s-framerate_comparision.pdf'))
plt.close()
