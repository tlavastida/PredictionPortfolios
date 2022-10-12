# PredictionPortfolios

Code for the paper "Faster Matching via Learned Duals" which will appear in NeurIPS 2022.  
Authors: Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, and Sergei Vassilvitskii  
(Openreview link to be added once available)

Code is provided as is.

See the author response discussion and Appendix A of the paper (link to be added once the Openreview page is up) for details on the experiments.  Datasets can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

## Overview

See below for a description of the code contained in each file.  The first group of files have been adapted from [the code](https://github.com/tlavastida/LearnedDuals) that was used for the experimental section in "Faster Matchings via Learned Duals" by Dinitz et al. NeurIPS 2021 [(Openreview link)](https://openreview.net/forum?id=kB8eks2Edt8).  The second group of files were created for this project.

### Code adapted from tlavastida/LearnedDuals

| File name | Description |
| ----------- | ----------- |
| BipartiteGraph.py | Bipartite graph data structure |
| MaxBipartiteMatching.py | Maximum cardinality matching in bipartite graphs via Hopcroft-Karp |
| MinWeightPerfectMatching.py | Minimum weight perfect matching in bipartite graphs via the Hungarian algorithm.  Also supports initialization with user-provided dual variables. |
| UtilityFunctions.py | Utility code that is used in several places |
| geometric_type_model_exp.py | Used to generate instances derived from geometric data |


### New code for this project

| File name | Description |
| ----------- | ----------- |
| KMedian.py | Clustering code used to generate sets of k predicted dual solutions |
| geometric_portfolio_experiment.py | Main file for running the experiments |

### Dependencies

- Numpy
- Scikit-learn (for k-means clustering)