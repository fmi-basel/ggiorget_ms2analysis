# Let's start with loading all the required packages
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

# Load data
path = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/benchmarking/spotdetection/trackpy/comparison'
# trackpy data
filenames = glob(os.path.join(path, '*-sweep.csv'))

# Load data
df = []
for filename in filenames:
    data = pd.read_csv(filename)
    df.append(data)
df = pd.concat(df)
df['clone'] = df['filename'].str.rsplit(pat='_', n=-1, expand=True)[3]

df = df[df['frame']<=600]
# Calculate precision, recall and f1 score
def calculate_scores(dataframe):
    precision = precision_score(dataframe['groundtruth'], dataframe['spot_detected'])
    recall = recall_score(dataframe['groundtruth'], dataframe['spot_detected'])
    f1 = f1_score(dataframe['groundtruth'], dataframe['spot_detected'])
    return f1, recall, precision


df_results = df.groupby(['threshold', 'spotdiameter', 'spot_size_max', 'spot_size_min']).apply(calculate_scores)
df_results = pd.DataFrame(df_results.tolist(), index=df_results.index,
                          columns=['f1', 'recall', 'precision']).reset_index()

spotdiameter = df['spotdiameter'].unique()

#df_results.to_csv(os.path.join(path, 'filtered_results.csv'), index=False)
# Plot results
# F1 score
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_results[df_results['spotdiameter'] == spotdiameter[i]]['threshold'].astype(str),
           df_results[df_results['spotdiameter'] == spotdiameter[i]]['f1'])
    ax.set_title(f"spotdiameter {spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.95, c='red', linestyle='dotted')
    ax.set_ylabel('F1 score')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'f1.pdf'))
plt.close()


# Recall
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_results[df_results['spotdiameter'] == spotdiameter[i]]['threshold'].astype(str),
           df_results[df_results['spotdiameter'] == spotdiameter[i]]['recall'])
    ax.set_title(f"spotdiameter {spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.95, c='red', linestyle='dotted')
    ax.set_ylabel('Recall')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'recall.pdf'))
plt.close()

# Precision
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    ax.bar(df_results[df_results['spotdiameter'] == spotdiameter[i]]['threshold'].astype(str),
           df_results[df_results['spotdiameter'] == spotdiameter[i]]['precision'])
    ax.set_title(f"spotdiameter {spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.95, c='red', linestyle='dotted')
    ax.set_ylabel('Precision')
ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'precision.pdf'))
plt.close()

# Plot results for each clone
df_results = df.groupby(['spotdiameter', 'threshold', 'clone']).apply(calculate_scores)
df_results = pd.DataFrame(df_results.tolist(), index=df_results.index,
                          columns=['f1', 'recall', 'precision']).reset_index()

# F1 score
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='f1', hue='clone',
                ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'f1_clone.pdf'))
plt.close()

# recall
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='recall', hue='clone',
                ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('Recall')
    ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'recall_clone.pdf'))
plt.close()

# precision
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='precision',
                hue='clone', ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('Precision')
    ax.set_xlabel('threshold')
plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'precision_clone.pdf'))
plt.close()

# Plot results for binned time point
# Plot results for each clone
df_results = df.groupby(['spotdiameter', 'threshold', pd.cut(df.frame, 12)]).apply(calculate_scores)
df_results = pd.DataFrame(df_results.tolist(), index=df_results.index,
                          columns=['f1', 'recall', 'precision']).reset_index()

# F1 score
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='f1', hue='frame',
                ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('F1 score')
    ax.set_xlabel('threshold')
    ax.get_legend().set_visible(False)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'f1_frame.pdf'))
plt.close()

# recall
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='recall', hue='frame',
                ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('Recall')
    ax.set_xlabel('threshold')
    ax.get_legend().set_visible(False)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'recall_frame.pdf'))
plt.close()

# precision
fig, axs = plt.subplots(len(spotdiameter), 1, figsize=(5, len(spotdiameter) * 3), sharey=True)
for i, ax in enumerate(axs.ravel()):
    sns.barplot(data=df_results[df_results['spotdiameter'] == spotdiameter[i]], x='threshold', y='precision',
                hue='frame', ax=ax)
    ax.set_title(f"{spotdiameter[i]}")
    ax.tick_params('x', labelrotation=45)
    # ax.axhline(y=0.73, c='red', linestyle='dotted')
    ax.set_ylabel('Precision')
    ax.set_xlabel('threshold')
    ax.get_legend().set_visible(False)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(path.replace('comparison', 'plots'), 'precision_frame.pdf'))
plt.close()
