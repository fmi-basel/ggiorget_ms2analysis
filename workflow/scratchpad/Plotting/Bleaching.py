# Load all the packages
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data
five_clone_name = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/summarized_tracks/fiveclones/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined.csv'
df_clones = pd.read_csv(five_clone_name, dtype={'clone': 'str'})
halfen_name = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_HalfEnhancer_combined.csv'
df_halfen = pd.read_csv(halfen_name, dtype={'clone': 'str'})

path_output = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/plotting_figures_movies/bleaching'

# Bleaching over time: average intensity as function of frame
average_int_clones = df_clones.groupby('frame')[['mean_spot', 'mean_localbackground', 'mean_completecell']].mean().reset_index()
average_int_halfenh = df_halfen.groupby('frame')[['mean_spot', 'mean_localbackground', 'mean_completecell']].mean().reset_index()

sns.lineplot(x='frame', y='mean_spot', data=average_int_clones, label='spot (including detected and background)')
sns.lineplot(x='frame', y='mean_localbackground', data=average_int_clones, label='local background')
sns.lineplot(x='frame', y='mean_completecell', data=average_int_clones, label='cell')
plt.ylabel('Average Intensity')
plt.xlim(-10, 600)
plt.title('five clones dataset')
plt.show()
#plt.savefig(os.path.join(path_output, 'fiveclones_avgintensity.pdf'))
plt.close()

sns.lineplot(x='frame', y='mean_spot', data=average_int_halfenh, label='spot (including detected and background)')
sns.lineplot(x='frame', y='mean_localbackground', data=average_int_halfenh, label='local background')
sns.lineplot(x='frame', y='mean_completecell', data=average_int_halfenh, label='cell')
plt.ylabel('Average Intensity')
plt.xlim(-10, 600)
plt.title('half enhancer dataset')
plt.show()
#plt.savefig(os.path.join(path_output, 'halfenh_avgintensity.pdf'))
plt.close()

# Bleaching over time: average intensity as function of frame
average_sd_clones = df_clones.groupby('frame')[['sd_spot', 'sd_localbackground', 'sd_completecell']].mean().reset_index()
average_sd_halfenh = df_halfen.groupby('frame')[['sd_spot', 'sd_localbackground', 'sd_completecell']].mean().reset_index()

sns.lineplot(x='frame', y='sd_spot', data=average_sd_clones, label='spot')
sns.lineplot(x='frame', y='sd_localbackground', data=average_sd_clones, label='local background')
sns.lineplot(x='frame', y='sd_completecell', data=average_sd_clones, label='cell')
plt.ylabel('Average SD')
plt.xlim(-10, 600)
plt.title('five clones dataset')
plt.show()
#plt.savefig(os.path.join(path_output, 'fiveclones_sd.pdf'))
plt.close()

sns.lineplot(x='frame', y='sd_spot', data=average_sd_halfenh, label='spot')
sns.lineplot(x='frame', y='sd_localbackground', data=average_sd_halfenh, label='local background')
sns.lineplot(x='frame', y='sd_completecell', data=average_sd_halfenh, label='cell')
plt.ylabel('Average SD')
plt.xlim(-10, 600)
plt.title('half enhancer dataset')
plt.show()
#plt.savefig(os.path.join(path_output, 'halfenh_avgintensity.pdf'))
plt.close()

# SNR over time
df_clones['SNR'] = df_clones['mean_spot']/df_clones['mean_localbackground']
average_snr_clones = df_clones[df_clones['spotdetected']==True].groupby('frame')[['mean_spot', 'mean_localbackground', 'SNR']].mean().reset_index()
df_halfen['SNR'] = df_halfen['mean_spot']/df_halfen['mean_localbackground']
average_snr_halfenh = df_halfen[df_halfen['spotdetected']==True].groupby('frame')[['mean_spot', 'mean_localbackground', 'SNR']].mean().reset_index()

sns.lineplot(x='frame', y='SNR', data=average_snr_clones, label='five clones dataset')
sns.lineplot(x='frame', y='SNR', data=average_snr_halfenh, label='half enhancer dataset')
plt.title('SNR')
plt.ylabel('SNR (spot/local background)')
plt.xlim(-10, 600)
plt.show()
#plt.savefig(os.path.join(path_output, 'snr.pdf'))
plt.close()

sns.lineplot(x='frame', y='mean_spot', data=average_snr_clones, label='spot')
sns.lineplot(x='frame', y='mean_localbackground', data=average_snr_clones, label='local background')
plt.ylabel('Average Intensity (spots only)')
plt.title('five clones dataset')
plt.xlim(-10, 600)
plt.show()
#plt.savefig(os.path.join(path_output, 'fiveclones_avgintensity_spots.pdf'))
plt.close()

sns.lineplot(x='frame', y='mean_spot', data=average_snr_halfenh, label='spot')
sns.lineplot(x='frame', y='mean_localbackground', data=average_snr_halfenh, label='local background')
plt.ylabel('Average Intensity (spots only)')
plt.title('half enhancer dataset')
plt.xlim(-10, 600)
plt.show()
#plt.savefig(os.path.join(path_output, 'halfenh_avgintensity_spots.pdf'))
plt.close()
