# FCCA

This repository contains the experimental code for the paper <a href="https://arxiv.org/abs/2211.09894">Features Compression based on Counterfactual Analysis</a>

## Installation

The MILP problem for computing the Counterfactual Explanation for a point is implemented in <a href="https://ampl.com/">AMPL</a> via a <a href="https://www.gurobi.com/solutions/gurobi-optimizer/?campaignid=18262689303&adgroupid=138243449982&creative=620260718865&keyword=gurobi&matchtype=e&gclid=Cj0KCQiA4OybBhCzARIsAIcfn9mYA1eyslmYMVKkmSzUWuZeLKwpNXdPrcIoKLnEr60zcnHFDSpc5j8aAgzgEALw_wcB">gurobi</a> solver. The directory where AMPL is installed has to be set in the configuration file *config.py* through the attribute *ampldir*.

The file *config.py* contains the configuration parameters of the project. 

## Configuration

The parameters that can modified via the configuration file *config.py*:

* *name*: name of the dataset. The dataset file is expected to be found in *datasets/{name}.csv*
* *lambda0, lambda1, lambda2*: hyperparameters for the Counterfactual Explanation objective function. They denote respectively the weight of the l0-, l1- and l2- norm.
* *p0, p1*: respectively the lower and upper bound for the classification probability of points in $\cal M$ (i.e. the points of the dataset for which it is required to compute the Counterfactual Explanation). §p0* is usually set to 0.5, meaning that no lower bound is required.
* *p2*: down-sampling percentage of $\cal M$.
* *CE_timelim*: timelimit (expressed in seconds) for the Counterfactual Explanation problem.
* *initializeCounterfactual*: boolean, set to True for initializing the Counterfactual Explanation solution by using the point $(\bar x, \bar y)\in{\cal D}_{tr}\colon \bar y = k^*$ with minimal euclidean distance from $x^0$.
* *d*: max_depth for the Decision Tree to build with the discretized dataset.
* *target_type*: the class of the Target black box model to train. Implemented: RandomForestClassifier.
* *target_parameters*: dictionary containing the training parameters of the Target black box model.
* *min_seed, max_seed*: random seeds for the experiments.
* *big_dataset*: boolean, if it's set to True we define the best quartile as the one that maximizes the compression rate without losing more than 1.5% of accuracy w.r.t. the baseline model; if it's set to False the best quartile is the one that maximizes the accuracy of the surrogate model.

For setting the parameters of a given dataset *dataset_name* we can call the function *set_name(dataset_name)*.

## Results

The output of the experiments can be found in folder *results/{dataset_name}*, where:
* Folder *seed* will contain all the output files of a single experiment with random seed equal to *seed*.
* Folder *dataset_name* will contain the overall output files of all the experiments performed.
