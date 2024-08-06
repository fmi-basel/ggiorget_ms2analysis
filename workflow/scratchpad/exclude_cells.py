import os

import numpy as np
import pandas as pd

# spotdetected_filtered
# Load data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/'
filename_2600 = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2600.csv'
excluded_2600 = 'fiveclones_excludedcells_threshold2600_truncated.csv'

df_2600 = pd.read_csv(os.path.join(path, filename_2600), dtype={'clone': 'str'})
df_excluded_2600 = pd.read_csv(os.path.join(path, excluded_2600))[['filename', 'track_id']]

df_2600 = df_2600.merge(df_excluded_2600, how='left', indicator=True)
df_2600['spotdetected_filtered_curated'] = df_2600.loc[:, 'spotdetected_filtered']
df_2600.loc[df_2600['_merge'] == 'both', 'spotdetected_filtered_curated'] = np.nan
df_2600 = df_2600.drop('_merge', axis=1)

# set promoter only cell line to all false
df_2600.loc[df_2600['clone'] == '6G3', 'spotdetected_filtered_curated'] = False

df_2600.to_csv(os.path.join(path, filename_2600.replace('.csv', '_curated.csv')), index=False)

# spotdetected
# Load data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/'
filename_2800 = '306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_30s_fiveclone_combined_threshold2800.csv'
excluded_2800 = 'fiveclones_excludedcells_threshold2800_truncated.csv'

df_2800 = pd.read_csv(os.path.join(path, filename_2800), dtype={'clone': 'str'})
df_excluded_2800 = pd.read_csv(os.path.join(path, excluded_2800))[['filename', 'track_id']]

# exclude cells from manually annotated data
df_2800 = df_2800.merge(df_excluded_2800, how='left', indicator=True)
df_2800['spotdetected_curated'] = df_2800.loc[:, 'spotdetected']
df_2800.loc[df_2800['_merge'] == 'both', 'spotdetected_curated'] = np.nan
df_2800['spotdetected_filtered_curated'] = df_2800.loc[:, 'spotdetected_filtered']
df_2800.loc[df_2800['_merge'] == 'both', 'spotdetected_filtered_curated'] = np.nan
df_2800 = df_2800.drop('_merge', axis=1)

# set promoter only cell line to all false
df_2800.loc[df_2800['clone'] == '6G3', 'spotdetected_curated'] = False
df_2800.loc[df_2800['clone'] == '6G3', 'spotdetected_filtered_curated'] = False

df_2800.to_csv(os.path.join(path, filename_2800.replace('.csv', '_curated.csv')), index=False)
