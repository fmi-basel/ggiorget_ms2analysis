# load packages
import os
from glob import glob

# name the directories where the projected data is stored:
# Five clones
dir = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/processed'
folder = [
    '20230714',
    '20230715',
    '20230716',
    '20230717',
    '20230718',
    '20230719',
    '20230720',
    '20230721',
    '20230722',
    '20230723',
    '20230724',
    '20230727',
]

# Half enhancer
dir = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/processed'
folder = [
    '20240131',
    '20240201',
    '20240202',
    '20240203',
    '20240204',
]

# create a list with all tiff files that are saved in those folders:
data_list = [glob(os.path.join(dir, folder, 'proj', '*_MAX.tiff')) for folder in folder]
data_list = [item for sublist in data_list for item in sublist]

# save the list as a txt file:
dir_out = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/E10_Mobi/live_imaging/data/helperlists'

with open(os.path.join(dir_out, 'fiveclone_dataset_list.txt'), 'w') as f:
    for item in data_list:
        f.write("%s\n" % item)
