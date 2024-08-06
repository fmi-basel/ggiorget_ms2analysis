# Installation
This project runs on Snakemake version 8.4.12. To install Snakemake, you can use conda. 
First create an environment and activate it:

```bash 
conda create -n snakemake
conda activate snakemake
``` 
Then install Snakemake:

```bash
conda install -c conda-forge -c bioconda snakemake=8.4.12
``` 

All other package dependencies are handles directly by Snakemake.

# Usage
To run the workflow, you need to provide a configuration file and a text file listing the samples to be processed.

The text file should contain the sample names (one per line), end with the extension '_dataset_list.txt', and be located in the data folder. An example is given.

The configuration file (config.yaml) should contain the following entries:
- movies_list: fill name of the text file containing the list of samples to be processed
- min_tracklength_segmentation: minimum length in frames a cell is visible 
- min_cellsize_segmentation: minimum size in pixels a cell has to have
- spotdiameter: diameter of the spots in pixels for spot detection
- spotdetection_threshold: threshold for spot detection
- spotfilter_size_min: minimum size of the spot for filtering after spot detection
- spotfilter_size_max: maximum size of the spot for filtering after spot detection
- spotfilter_mass: maximum mass of the spot for filtering after spot detection
- min_burstlength: filter on the minimum length of a transcriptional on time
An example is provided in the config folder.

To execute the workflow, navigate to the root of this project and run the following command in the activates snakemake environment:

```bash
snakemake --use-conda
```

It is possible to run the workflow in parallel by providing the number of cores to use:

```bash
snakemake --use-conda --cores 4
```
