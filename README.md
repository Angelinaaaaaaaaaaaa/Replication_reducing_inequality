# Addressing Voter Turnout Disparities through Data-Driven Resource Allocation
DSC180 Project by Cici and Angelina
# WI25


# FA24
# paper1: Replication-Active-learning-for-optimal-intervention-design-in-causal-models
DSC180 Project by Cici and Angelina 

This is an replication for paper: Active learning for optimal intervention design in causal models (Nature Machine Intelligence, 2023). Specifically, we focused on replicating the experiment on synthetic data. 

## Installation
Follow the two steps illustrated below

1. create a conda environment using `environment.yaml` (all dependencies are included; whole process takes about 5 min):
```
cd paper1
conda env create -f environment.yml
```
2. install the current package in editable mode inside the conda environment:
```
pip install -e .
```

## Synthetic data
Run on a synthetic instance, e.g.:
```
python run.py --nnodes 5 --noise_level 1 --DAG_type path --std --a_size 2 --a_target 3 4 --acquisition greedy
```
This runs a synthetic data experiment using 1 specified acqusition function. 
After running, results will be stored in a folder, which contains of 3 pickle files. 

To visualze the results: 
```
python visualze_results.py
```
More examples given in: paper1//optint/notebook/test_multigraphs.ipynb
Visualization of results comparison are stored in `paper1//optint/results`

Source code folder: `paper1//optint/`

# paper2: reducing_inequality
## Overview
This repository contains the code and materials for the project "Reducing Inequality through Optimal Intervention Design", which integrates two complementary approaches for intervention design:

`Active Learning for Optimal Intervention Design in Causal Models`
This approach uses active learning within causal models to efficiently design interventions that shift a system's mean toward a desired target with minimal intervention effort.

`Making Decisions that Reduce Discriminatory Impact`
This approach incorporates fairness constraints into causal optimization, ensuring that interventions maximize societal benefit while addressing systemic inequities across protected groups.

The repository contains the implementation of these frameworks, supporting datasets, and the scripts to reproduce the results described in the report.

Repository for "Disaggregated Interventions to Reduce Inequality."

arXiv: <https://arxiv.org/abs/2107.00593>

Includes supporting code and data to reproduce figures and results. 
To generate figures and print results from the paper, run `plot_results.ipynb`.
To re-run optimization problems, run `optimize_disaggregated.py` and `optimize.py` in that order.
New result files (`.npy`) will be output upon re-running. Note that re-running is slow.
Additional documentation and significant code restructuring forthcoming!



