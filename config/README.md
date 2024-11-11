# Installation
This project runs on Snakemake version 8.4.12. To install Snakemake, you can use Mambaforge. 
First create an environment and activate it:

```bash 
mamba create -n snakemake
mamba activate snakemake
```
Then install Snakemake:

```bash
mamba install -c conda-forge -c bioconda snakemake=8.4.12
``` 

All other package dependencies are handles directly by Snakemake.

# Usage
## Pre-processing before using snakemake
The snakemake workflow processes max- and mean-projections (xyt) from time-resolved 3D-stacks (xyzt). Before running the snakemake workflow, these projection files can be generated with scripts provided in the folder [scripts](../workflow/scripts/s00_preprocessing).  

MS2 data in form from *.nd files are processed as max-projections with the script [max-proj_nd-as-tiff.py](../workflow/scripts/s00_preprocessing/max-proj_nd-as-tiff.py). One file corresponds to z-stack movies of several stage positions (direct output from the microscope).

```bash
python max-proj_nd-as-tiff.py path/to/images.nd path/to/outputfolder
``` 

GFP data in form of *.nd files are processed as mean-projections with the script [mean-proj_nd-as-tiff.py](../workflow/scripts/s00_preprocessing/mean-proj_nd-as-tiff.py). One file corresponds to single z-stacks of movie-corresponding stage positions.

```bash
python mean-proj_nd-as-tiff.py path/to/images.nd path/to/outputfolder
```

Output folder strcture should be as follows: data/processed/{date_of_aquisition}/proj/

## Snakemake workflow
To run the workflow, you need to provide a configuration file and a text file listing the samples to be processed (list of max-projections).

The text file should contain the sample names (one per line), end with the extension '_dataset_list.txt', and be located in the data folder. An example is given.

The configuration file should contain the following entries:
- movies_list: fill name of the text file containing the list of samples to be processed
- min_tracklength_segmentation: minimum length in frames a cell is visible 
- min_cellsize_segmentation: minimum size in pixels a cell has to have
- spotdiameter: diameter of the spots in pixels for spot detection
- spotdetection_threshold: threshold for spot detection
- spotfilter_size_min: minimum size of the spot for filtering after spot detection
- spotfilter_size_max: maximum size of the spot for filtering after spot detection
- spotfilter_mass: maximum mass of the spot for filtering after spot detection
- min_burstlength: filter on the minimum length of a transcriptional on time

An example is provided in the config folder (example_config.yaml). If you rename the configuration file, make sure to adjust the name in the Snakefile.

To dry-run the workflow (-np), navigate to the root of this project and run the following command in the activated snakemake environment:

```bash
snakemake --use-conda --cores 1 -np
```

To run the pipeline, remove the -np flag:

```bash
snakemake --use-conda --cores 1 
```
