# Using the Benchmarking_spotdetection.py script, I want to compare different spot detection algorithms and
# pre-processing approaches. Here, I plot the resulting statistics and calculate various rates, to decide on a good
# compromise between detecting spots and false-positives. In detail, I want to find a setting with a high sensitivity,
# while keeping the false discovery rate (FDR) low. Differently speaking I'd like to find a setting with a high F1 score.


# Let's start with loading all the required packages
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
path = '/Volumes/ggiorget_scratch/Jana/Microscopy/Mobilisation-E10/groundtruth/'
# trackpy data
filename_trackpy1 = '20230715_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5G7_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_cutMAX_summary_spotdetec_trackpy_sub.csv'
filename_trackpy2 = '20230719_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5E10_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s1_MAX_summary_spotdetec_trackpy_sub.csv'
filename_trackpy3 = '20230720_306KI-ddCTCF-dpuro-MS2-HaloMCP-E10Mobi_JF549_5F11_1_FullseqTIRF-mCherry-GFPCy5WithSMB_s5_MAX_summary_spotdetec_trackpy_sub.csv'
df_1 = pd.read_csv(os.path.join(path, filename_trackpy1))
df_1['clone'] = filename_trackpy1.split('_')[3]
df_2 = pd.read_csv(os.path.join(path, filename_trackpy2))
df_2['clone'] = filename_trackpy2.split('_')[3]
df_3 = pd.read_csv(os.path.join(path, filename_trackpy3))
df_3['clone'] = filename_trackpy3.split('_')[3]
# combine data from 3 movies
df_trackpy = df_1.drop(['diameter', 'threshold', 'images', 'clone'], axis=1) + df_2.drop(
    ['diameter', 'threshold', 'images', 'clone'],
    axis=1) + df_3.drop(
    ['diameter', 'threshold', 'images', 'clone'],
    axis=1)
df_trackpy = df_trackpy.merge(df_1[['diameter', 'threshold', 'images']], left_index=True, right_index=True)
# calculate rates
df_trackpy['FDR'] = df_trackpy['FP'] / (df_trackpy['FP'] + df_trackpy['TP'])
df_trackpy['Precision'] = df_trackpy['TP'] / (df_trackpy['FP'] + df_trackpy['TP'])
df_trackpy['Sensitivity'] = df_trackpy['TP'] / (df_trackpy['FN'] + df_trackpy['TP'])
df_trackpy['F1score'] = 2 * df_trackpy['TP'] / (2 * df_trackpy['TP'] + df_trackpy['FP'] + df_trackpy['FN'])

"""
filename1_hmax = filename1.replace('_tracks.csv', '_summary_spotdetec_hmax.csv')
filename2_hmax = filename2.replace('_tracks.csv', '_summary_spotdetec_hmax.csv')
df_1 = pd.read_csv(os.path.join(path_groundtruth, filename1_hmax))
df_2 = pd.read_csv(os.path.join(path_groundtruth, filename2_hmax))
df_hmax = df_1.drop(['sd_threshold', 'images'], axis=1) + df_2.drop(['sd_threshold', 'images'], axis=1)
df_hmax = df_hmax.merge(df_1[['sd_threshold', 'images']], left_index=True, right_index=True)
df_hmax['FDR'] = df_hmax['FP'] / (df_hmax['FP'] + df_hmax['TP'])
df_hmax['Precision'] = df_hmax['TP'] / (df_hmax['FP'] + df_hmax['TP'])
df_hmax['Sensitivity'] = df_hmax['TP'] / (df_hmax['FN'] + df_hmax['TP'])
df_hmax['F1score'] = 2 * df_hmax['TP'] / (2 * df_hmax['TP'] + df_hmax['FP'] + df_hmax['FN'])
"""

# -- Trackpy Plotting --
# For visualisation I look at rates independently dependent on the diameter in detection
diameter = 9
df_trackpy_dia = df_trackpy[df_trackpy['diameter'] == diameter]
projection = np.unique(df_trackpy_dia['images']).tolist()


# TP count
fig, axs = plt.subplots(len(projection), 1, figsize=(5, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['threshold'].astype(str),
           df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['TP'])
    ax.set_title(f"{projection[i]}")
    ax.axhline(y=180, c='red', linestyle='dotted')
    ax.set_ylabel('True positive count')
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plots_spotdetection/trackpy_TP_count.pdf'))
plt.close()

# precision
fig, axs = plt.subplots(len(projection), 1, figsize=(5, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['threshold'].astype(str),
           df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['Precision'])
    ax.set_title(f"{projection[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.95, c='red', linestyle='dotted')
    ax.set_ylabel('Precision')
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_precision.pdf'))
plt.close()

# Sensitivity
fig, axs = plt.subplots(len(projection), 1, figsize=(5, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['threshold'].astype(str),
           df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['Sensitivity'])
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.7, c='red', linestyle='dotted')
    ax.set_ylabel('Sensitivity')
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_sensitivity.pdf'))
plt.close()

# F1 score
fig, axs = plt.subplots(15, 1, figsize=(5, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['threshold'].astype(str),
           df_trackpy_dia[df_trackpy_dia['images'] == projection[i]]['F1score'])
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('F1score')
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_f1.pdf'))
plt.close()

# plot a heatmap where the f1 values are encoded in color from the df_trackpy dataframe with threshold on one axis and projection on the other
# create a pivot table
df_trackpy_pivot = df_trackpy_dia.pivot(index='threshold', columns='images', values='TP')
df_trackpy_pivot = df_trackpy_pivot.drop(['Maxprojection_corr', 'Maxprojection_corr_backgroundsub3','Maxprojection_corr_backgroundsub5','Maxprojection_corr_backgroundsub7','Maxprojection_corr_backgroundsub9'], axis=1)
df_trackpy_pivot.dropna(axis=0, how='all', inplace=True)
projection.remove('Maxprojection_corr')
projection.remove('Maxprojection_corr_backgroundsub3')
projection.remove('Maxprojection_corr_backgroundsub5')
projection.remove('Maxprojection_corr_backgroundsub7')
projection.remove('Maxprojection_corr_backgroundsub9')
projection_names = projection.copy()
for i in range(len(projection_names)):
    projection_names[i] = projection_names[i].replace('Maxprojection', 'Max')
    projection_names[i] = projection_names[i].replace('_backgroundsub', ' BG_')
    projection_names[i] = projection_names[i].replace('_corr', ' FFC')
    projection_names[i] = projection_names[i].replace('_ratio', ' bleach')

# plot
plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 20})
plt.imshow(df_trackpy_pivot, cmap='viridis', aspect='auto', interpolation='none')
plt.xticks(np.arange(0, len(projection)), projection_names, rotation=45, ha='right', va='top')
plt.yticks(np.arange(0, len(df_trackpy_pivot.index)), df_trackpy_pivot.index)
plt.ylabel('threshold')
plt.colorbar()
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/JF549/trackpy_tp_heatmap.svg'), format='svg')
plt.close()



"""
# hmax Plotting
# TP count
projection = np.unique(df_hmax['images']).tolist()
fig, axs = plt.subplots(5, 1, figsize=(5, 15), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_hmax[df_hmax['images'] == projection[i]]['sd_threshold'].astype(str),
           df_hmax[df_hmax['images'] == projection[i]]['TP'])
    ax.set_title(f"{projection[i]}")
    ax.axhline(y=180, c='red', linestyle='dotted')
    ax.set_ylabel('True positive count')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plots_spotdetection/hmax_TP_count.pdf'))
plt.close()

# False discovery rate
fig, axs = plt.subplots(5, 1, figsize=(5, 15), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_hmax[df_hmax['images'] == projection[i]]['sd_threshold'].astype(str),
           df_hmax[df_hmax['images'] == projection[i]]['FDR'])
    ax.set_title(f"{projection[i]}")
    ax.tick_params('x', labelrotation=45)
    ax.axhline(y=0.95, c='red', linestyle='dotted')
    ax.set_ylabel('FDR')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plots_spotdetection/hmax_FDR.pdf'))
plt.close()

# Sensitivity
fig, axs = plt.subplots(5, 1, figsize=(5, 15), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_hmax[df_hmax['images'] == projection[i]]['sd_threshold'].astype(str),
           df_hmax[df_hmax['images'] == projection[i]]['Sensitivity'])
    ax.set_title(f"{projection[i]}")
    ax.axhline(y=0.6, c='red', linestyle='dotted')
    ax.set_ylabel('Sensitivity')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plots_spotdetection/hmax_sensitivity.pdf'))
plt.close()

# F1 score
fig, axs = plt.subplots(5, 1, figsize=(5, 15), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_hmax[df_hmax['images'] == projection[i]]['sd_threshold'].astype(str),
           df_hmax[df_hmax['images'] == projection[i]]['F1score'])
    ax.set_title(f"{projection[i]}")
    ax.axhline(y=0.14, c='red', linestyle='dotted')
    ax.set_ylabel('F1score')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plots_spotdetection/hmax_f1.pdf'))
plt.close()
"""

# -- Rates per clone --
# create a new dataframe with all the data including info on clone
df_trackpy_clone = pd.concat([df_1, df_2, df_3])
# calculate rates
df_trackpy_clone['Precision'] = df_trackpy_clone['TP'] / (df_trackpy_clone['FP'] + df_trackpy_clone['TP'])
df_trackpy_clone['Sensitivity'] = df_trackpy_clone['TP'] / (df_trackpy_clone['FN'] + df_trackpy_clone['TP'])
df_trackpy_clone['F1score'] = 2 * df_trackpy_clone['TP'] / (
            2 * df_trackpy_clone['TP'] + df_trackpy_clone['FP'] + df_trackpy_clone['FN'])

# For visualisation, I look at rates dependent on the diameter in detection
diameter = 9
df_trackpy_clone_dia = df_trackpy_clone[df_trackpy_clone['diameter'] == diameter]

projection = np.unique(df_trackpy_clone['images']).tolist()
clones = np.unique(df_trackpy_clone['clone']).tolist()

# F1 score
fig, axs = plt.subplots(len(projection), 1, figsize=(10, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'threshold'] - 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'F1score'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'threshold'],
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'F1score'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'threshold'] + 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'F1score'], width=50)
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('F1score')
    ax.set_xticks(df_trackpy_clone_dia[
                      (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
                      'threshold'])
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.legend(clones)
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_f1_clones.pdf'))
plt.close()

# Precision
fig, axs = plt.subplots(len(projection), 1, figsize=(10, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'threshold'] - 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'Precision'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'threshold'],
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'Precision'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'threshold'] + 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'Precision'], width=50)
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('Precision')
    ax.set_xticks(df_trackpy_clone_dia[
                      (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
                      'threshold'])
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.legend(clones)
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_precision_clones.pdf'))
plt.close()

# Recall
fig, axs = plt.subplots(len(projection), 1, figsize=(10, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'threshold'] - 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'Sensitivity'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'threshold'],
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'Sensitivity'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'threshold'] + 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'Sensitivity'], width=50)
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('Sensitivity')
    ax.set_xticks(df_trackpy_clone_dia[
                      (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
                      'threshold'])
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.legend(clones)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path, 'plot_benchmark_spotdetection/fiveclones/trackpy_sensitivity_clones.pdf'))
plt.close()

# FP
fig, axs = plt.subplots(len(projection), 1, figsize=(10, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'threshold'] - 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'FP'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'threshold'],
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'FP'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'threshold'] + 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'FP'], width=50)
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('FP')
    ax.set_xticks(df_trackpy_clone_dia[
                      (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
                      'threshold'])
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.legend(clones)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plot_benchmark_spotdetection/fiveclones/trackpy_f1.pdf'))
plt.close()

# TP
fig, axs = plt.subplots(len(projection), 1, figsize=(10, len(projection)*3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'threshold'] - 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[0])][
               'TP'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'threshold'],
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[1])][
               'TP'], width=50)
    ax.bar(df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'threshold'] + 50,
           df_trackpy_clone_dia[
               (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
               'TP'], width=50)
    ax.set_title(f"{projection[i]}")
    # ax.axhline(y=0.55, c='red', linestyle='dotted')
    ax.set_ylabel('TP')
    ax.set_xticks(df_trackpy_clone_dia[
                      (df_trackpy_clone_dia['images'] == projection[i]) & (df_trackpy_clone_dia['clone'] == clones[2])][
                      'threshold'])
ax.set_xlabel('threshold')
fig.suptitle("Estimated diameter = {}".format(diameter), fontsize=16)
plt.legend(clones)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path_groundtruth, 'plot_benchmark_spotdetection/fiveclones/trackpy_f1.pdf'))
plt.close()
