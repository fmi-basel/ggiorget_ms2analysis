# Snakemake workflow: 'Enhancer control of transcriptional activity via modulation of burst frequency'

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/<owner>/<repo>/workflows/Tests/badge.svg?branch=main)](https://github.com/<owner>/<repo>/actions?query=branch%3Amain+workflow%3ATests)


A Snakemake workflow for live-cell image analysis workflow as described in detail in 'Enhancer control of transcriptional activity via modulation of burst frequency'

In short, the workflow performs following steps:
1. Cell segmentation using stardist and tracking of cells
2. Spot detection and filtering
3. Spot linking
4. Spot intensity read-out

For installation and usage of the workflow, please follow the instructions in the folder [config](config/README.md).

Minimal example data is provided, for the full dataset please contact the authors.
