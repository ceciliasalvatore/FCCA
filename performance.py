import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from config import cfg

def getPerformanceValue(performance_dict, type):
    if isinstance(performance_dict, Performance):
        return performance_dict.__getattribute__(type)
    elif isinstance(performance_dict[list(performance_dict.keys())[0]], Performance):
        values = np.zeros((len(performance_dict.keys()),1))
    elif isinstance(performance_dict[list(performance_dict.keys())[0]], dict):
        values = np.zeros((len(cfg.quantiles), len(performance_dict.keys())))

    for k in range(len(performance_dict.keys())):
        key = list(performance_dict.keys())[k]
        if isinstance(performance_dict[key],Performance):
            values[k,0] = performance_dict[key].__getattribute__(type)
        elif isinstance(performance_dict[key],dict):
            for q in range(len(cfg.quantiles)):
                quantile = cfg.quantiles[q]
                values[q,k] = performance_dict[key][quantile].__getattribute__(type)

    return values

class Performance:
    def __init__(self, model, compression=None, inconsistency=None):
        self.model = model
        self.accuracy = None
        self.confusion_matrix = None
        self.compression = compression
        self.inconsistency = inconsistency

        if isinstance(model, DecisionTreeClassifier):
            self.features_used = np.unique(model.tree_.feature[model.tree_.feature >= 0])
            self.n_features = len(self.features_used)
        if isinstance(model, RandomForestClassifier):
            self.features_used = np.array([])
            for t in model.estimators_:
                self.features_used = np.concatenate((self.features_used, t.tree_.feature[t.tree_.feature >= 0]))
            self.features_used = np.unique(self.features_used)
            self.n_features = len(self.features_used)


    def evaluate(self, x, y):
        y_pred = self.model.predict(x)
        self.accuracy = accuracy_score(y, y_pred)
        self.confusion_matrix = confusion_matrix(y, y_pred)

