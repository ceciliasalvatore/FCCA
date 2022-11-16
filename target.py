import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from joblib import load, dump
from CounterfactualExplanation import CE
from config import cfg
from scipy.spatial import distance_matrix
from utils import find_row_in_matrix, add_row_to_matrix

class TargetSystem:
    def __init__(self, dataset):
        self.dataset = dataset

        self.log_file_name = f"{cfg.getResultsSeedsDir()}/TargetSystemLog.txt"
        self.log_file = open(self.log_file_name, mode="a")

        self.counterfactualSolver = None

        model_dir = f"{cfg.getResultsSeedsDir()}/target_{cfg.getModelInfos()}.joblib"

        if os.path.exists(model_dir):
            self.model = load(model_dir)
        else:
            self.train(model_dir)
            self.evaluate()

    def train(self, model_dir):
        estimator = cfg.target_type()
        for k, v in cfg.target_parameters.items():
            estimator.__setattr__(k, v)
        params = {'max_depth':[3,4,6,8,10]}
        self.model = GridSearchCV(estimator=estimator, param_grid=params)
        self.model.fit(self.dataset.x_tr, self.dataset.y_tr)
        self.model = self.model.best_estimator_
        print("RANDOM FOREST", file=self.log_file)
        print(self.model, file=self.log_file)
        dump(self.model, model_dir)

    def evaluate(self):
        y_tr_pred = self.model.predict(self.dataset.x_tr)
        y_ts_pred = self.model.predict(self.dataset.x_ts)

        print("Train confusion matrix:", file=self.log_file)
        print(confusion_matrix(self.dataset.y_tr, y_tr_pred), file=self.log_file)
        print("Test confusion matrix:", file=self.log_file)
        print(confusion_matrix(self.dataset.y_ts, y_ts_pred), file=self.log_file)

        train_acc = accuracy_score(self.dataset.y_tr, y_tr_pred)
        test_acc = accuracy_score(self.dataset.y_ts, y_ts_pred)

        print("Train accuracy: {} - Test accuracy: {}".format(train_acc, test_acc), file=self.log_file)

        self.log_file.flush()

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def getCEs(self, x0, k):
        x0_filename = cfg.get_x_Filename('x0')
        xCE_filename = cfg.get_x_Filename('xCE')

        if os.path.exists(x0_filename):
            x0_pool = pd.read_csv(x0_filename).to_numpy()
            xCE_pool = pd.read_csv(xCE_filename).to_numpy()
            y0_pool = x0_pool[:, -1]
            x0_pool = x0_pool[:, :-1]
            yCE_pool = xCE_pool[:, -1]
            xCE_pool = xCE_pool[:, :-1]
        else:
            x0_pool = np.array([]).reshape((0, len(self.dataset.features_name)))
            y0_pool = np.array([])
            xCE_pool = np.array([]).reshape((0, len(self.dataset.features_name)))
            yCE_pool = np.array([])

        xCE = np.array([]).reshape((0, len(self.dataset.features_name)))
        yCE = np.array([])

        for i in range(len(x0)):
            pool_index = find_row_in_matrix(x0_pool, x0[i,:])
            if len(pool_index)>=1:
                pool_index = pool_index[0]
                xCE_i = xCE_pool[pool_index,:]
                yCE_i = [yCE_pool[pool_index]]
            else:
                xCE_i = self.CounterfactualExplanation(x0[i, :], k[i])
                yCE_i = self.predict(xCE_i.reshape((1,-1)))
                x0_pool = add_row_to_matrix(x0_pool, x0[i,:])
                xCE_pool = add_row_to_matrix(xCE_pool, xCE_i)
                y0_pool = np.concatenate((y0_pool, self.predict(x0[i,:].reshape((1,-1)))))
                yCE_pool = np.concatenate((yCE_pool, yCE_i))
                x0_df = pd.DataFrame(x0_pool, columns=self.dataset.features_name)
                x0_df['Y'] = y0_pool
                xCE_df = pd.DataFrame(xCE_pool, columns=self.dataset.features_name)
                xCE_df['Y'] = yCE_pool
                x0_df.to_csv(x0_filename, index=False)
                xCE_df.to_csv(xCE_filename, index=False)

            xCE = add_row_to_matrix(xCE, xCE_i)
            yCE = np.concatenate((yCE, yCE_i))

        return xCE, yCE

    def CounterfactualExplanation(self, x0, k):

        print("Computing counterfactual couple")
        if self.counterfactualSolver is None:
            self.counterfactualSolver = CE(self.model, self.dataset, self.log_file)

        if cfg.initializeCounterfactual:
            # initialPoint : point in the other class nearer to x'
            D = distance_matrix(x0.reshape((1,-1)), self.dataset.x_tr)
            D[:, self.model.predict(self.dataset.x_tr) != k] = 1000
            N = D.argsort(axis=1)
            initialPoint = self.dataset.x_tr[N[0, 0], :]
        else:
            initialPoint = None
        xCE = self.counterfactualSolver.solve(x0, k, initialPoint)

        return xCE