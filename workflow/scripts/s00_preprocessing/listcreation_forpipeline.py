# load packages
from glob import glob
import os


# name the directories where the projected data is stored:
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

# save the list as a txt file:
dir_out = '/Volumes/ggiorget_scratch/Jana/transcrip_dynamic/HalfEnhancer/live_imaging/data/helperlists'

with open(os.path.join(dir_out, 'HalfEnhancer_dataset_list.txt'), 'w') as f:
    for item in data_list:
        f.write("%s\n" % item)

