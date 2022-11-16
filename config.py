import os
from sklearn.ensemble import RandomForestClassifier

ampldir = "C:\\Users\\cecis\\Documents\\AMPL"

class Config:
    def __init__(self):

        self.data_base_dir = "datasets" # Directory for datasets
        self.results_base_dir = "results" # Directory for results

        # Dataset name
        self.set_name('boston')
        # self.set_name('arrhythmia')
        # self.set_name('ionosphere')
        # self.set_name('magic')
        # self.set_name('particle')
        # self.set_name('vehicle')

        self.initializeCounterfactual = True

        # AMPL Timelimits
        # if integer --> timelimit in seconds
        # if None --> no timelimit & set verbose
        self.CE_timelim = 40 * 60 # Time limit for the Counterfactual Explanation problem

        self.counterfactual_labels = {0:1, 1:0} # Label required for the Counterfactual Explanations
        self.n_decimals = 2 # Number of significative decimals for thresholds
        self.quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]


    def set_name(self, name):
        self.name = name
        self.data_dir = "{}/{}.csv".format(self.data_base_dir, self.name)

        if self.name == 'boston':
            self.p0 = .5
            self.p1 = 1
            self.p2 = 1
            self.lambda0 = 0.1
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 3
            self.test_split = 0.3
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 100, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 120
            self.seed = self.min_seed
            self.big_dataset = False

        elif self.name == 'arrhythmia':
            self.p0 = .5
            self.p1 = 1
            self.p2 = 1
            self.lambda0 = 0.1
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 4
            self.test_split = 0.2
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 100, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 120
            self.seed = self.min_seed
            self.big_dataset = False

        elif self.name == 'ionosphere':
            self.p0 = .5
            self.p1 = 1
            self.p2 = 1
            self.lambda0 = 0.1
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 3
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 100, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 120
            self.seed = self.min_seed
            self.big_dataset = False

        elif self.name == 'magic':
            self.p0 = .5
            self.p1 = .7
            self.p2 = .2
            self.lambda0 = 0.05
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 6
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 100, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 105
            self.seed = self.min_seed
            self.big_dataset = True

        elif self.name == 'particle':
            self.p0 = .5
            self.p1 = .7
            self.p2 = .2
            self.lambda0 = 0.1
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 4
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 30, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 105
            self.seed = self.min_seed
            self.big_dataset = True

        elif self.name == 'vehicle':
            self.p0 = .5
            self.p1 = .7
            self.p2 = .05
            self.lambda0 = 1
            self.lambda1 = .1
            self.lambda2 = 0
            self.d = 6
            self.max_seed = 105
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 30, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 105
            self.seed = self.min_seed
            self.big_dataset = True

        else:
            print("Unrecognized dataset name, setting standard parameters in Config")
            self.p0 = .5
            self.p1 = 1  # Max probability for x0 points
            self.p2 = 1  # Percentage of points to subsample x0 with KMeans
            self.lambda0 = 0.1
            self.lambda1 = 1
            self.lambda2 = 0
            self.d = 3
            self.target_type = RandomForestClassifier
            self.target_parameters = {'n_estimators': 100, 'class_weight': 'balanced'}
            self.min_seed = 100
            self.max_seed = 120
            self.seed = self.min_seed
            self.big_dataset = False

    def getResultsDir(self):
        results_dir = f"{self.results_base_dir}/{self.name}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir

    def getResultsSeedsDir(self):
        results_seed_dir = f"{self.results_base_dir}/{self.name}/{self.seed}"
        if not os.path.exists(results_seed_dir):
            os.makedirs(results_seed_dir)
        return results_seed_dir

    def get_x_Filename(self, type="x0"):
        return f"{self.getResultsSeedsDir()}/{type}_{self.lambda0}_{self.lambda1}.txt"

    def get_train_test_Filename(self, type='train'):
        return f"{self.getResultsSeedsDir()}/{type}.txt"

    def get_counterfactual_log(self):
        file = "{}/counterfactuals_log.txt".format(self.getResultsSeedsDir())
        return file

    def get_log_Filename(self):
        file = "{}/log.txt".format(self.getResultsDir())
        return file

    def get_seed_log_Filename(self):
        file = "{}/log.txt".format(self.getResultsSeedsDir())
        return file

    def getModelInfos(self):
        name = str(self.target_type).split("'")[1].split(".")[-1]
        for k,v in self.target_parameters.items():
            name = "{}_{}_{}".format(name, k, v)
        return name

    def getPlotDirectory(self, seed=False):
        dir = self.getResultsDir()
        if seed == True:
            dir = self.getResultsSeedsDir()
        dir = f"{dir}/{self.name}"
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

cfg = Config()