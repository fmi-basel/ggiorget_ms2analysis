# Snakemake workflow: 'Using Live-cell Imaging to investigate Enhancer-driven Transcriptional Dynamics'

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/<owner>/<repo>/workflows/Tests/badge.svg?branch=main)](https://github.com/<owner>/<repo>/actions?query=branch%3Amain+workflow%3ATests)


A Snakemake workflow for live-cell image analysis workflow as decribed in detail in <citation>

In short, the workflow performs following steps:
1. Cell segmentation using stardist and tracking of cells
2. Spot detection and filtering
3. Spot linking
4. Spot intensity read-out

For installation of the workflow, please follow the instructions in the [config](config/README.md).

Author:
Jana Tünnermann

# TODO

* Replace `<owner>` and `<repo>` everywhere in the template (also under .github/workflows) with the correct `<repo>` name and owning user or organization.
* Replace `<name>` with the workflow name (can be the same as `<repo>`).
* Replace `<description>` with a description of what the workflow does.
* The workflow will occur in the snakemake-workflow-catalog once it has been made public. Then the link under "Usage" will point to the usage instructions if `<owner>` and `<repo>` were correctly set.