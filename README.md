# reducing_inequality
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



