import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from config import cfg

class Dataset:
    def __init__(self, test_size=0.3):
        df = pd.read_csv(cfg.data_dir)
        features_name = df.columns.tolist()[:-1]
        features_type = df.iloc[0].tolist()[:-1]
        features_actionability = df.iloc[1].to_list()[:-1]

        x = df.iloc[2:].to_numpy().astype(float)
        y = x[:, -1].astype(float).astype(int)
        x = x[:, :-1]

        preprocessed_x = None
        self.features_name = None
        self.features_type = None
        self.features_actionability = None
        self.c_onehot = 0

        for j in range(len(features_name)):
            feature_type = features_type[j]
            if feature_type == "C":  # categorical
                enc = OneHotEncoder(sparse=False)
                feature = enc.fit_transform(x[:, j].reshape((-1, 1)))
                feature_name = enc.categories_[0].tolist()
                feature_type = ["C{}".format(self.c_onehot) for c in feature_name]
                feature_actionability = [features_actionability[j] for c in feature_name]
                self.c_onehot += 1
            else:
                feature_name = [features_name[j]]
                feature_type = [feature_type]
                feature_actionability = [features_actionability[j]]
                feature = x[:, j].astype(float).reshape((-1, 1))

            if preprocessed_x is None:
                preprocessed_x = feature
                self.features_name = feature_name
                self.features_type = feature_type
                self.features_actionability = feature_actionability
            else:
                preprocessed_x = np.concatenate((preprocessed_x, feature), axis=1)
                self.features_name += feature_name
                self.features_type += feature_type
                self.features_actionability += feature_actionability
        self.classes_ = np.unique(y)

        if os.path.exists(cfg.get_train_test_Filename('train')):
            data_tr = pd.read_csv(cfg.get_train_test_Filename('train'), header=None).to_numpy()
            data_ts = pd.read_csv(cfg.get_train_test_Filename('test'), header=None).to_numpy()
            self.x_tr = data_tr[:,:-1]
            self.y_tr = data_tr[:,-1]
            self.x_ts = data_ts[:,:-1]
            self.y_ts = data_ts[:,-1]
        else:
            self.x_tr, self.x_ts, self.y_tr, self.y_ts = train_test_split(preprocessed_x, y, test_size=test_size, stratify=y)

            scaler = MinMaxScaler()
            self.x_tr = scaler.fit_transform(self.x_tr)
            self.x_ts = scaler.transform(self.x_ts)

            self.y_ts = self.y_ts[np.all(self.x_ts>=-100, axis=1)]
            self.x_ts = self.x_ts[np.all(self.x_ts>=-100, axis=1),:]
            self.y_ts = self.y_ts[np.all(self.x_ts<=100, axis=1)]
            self.x_ts = self.x_ts[np.all(self.x_ts<=100, axis=1),:]

            pd.DataFrame(data=np.concatenate((self.x_tr, self.y_tr.reshape((-1,1))),axis=1)).to_csv(cfg.get_train_test_Filename('train'), index=False, header=False)
            pd.DataFrame(data=np.concatenate((self.x_ts, self.y_ts.reshape((-1,1))),axis=1)).to_csv(cfg.get_train_test_Filename('test'), index=False, header=False)

        self.lb = np.min(self.x_tr,axis=0).tolist()
        self.ub = np.max(self.x_tr,axis=0).tolist()

        self.eps = []
        for i in range(len(self.features_name)):
            try:
                self.eps.append(max(1.e-4, np.min(np.diff(np.unique(self.x_tr[:,i])))/2))
            except:
                self.eps.append(1.e-4)

    def getRelevant(self, x, y, k, model):
        index = np.where(
            (y == k) & (model.predict(x) == k) & (np.max(model.predict_proba(x), axis=1) >= cfg.p0) & (
                        np.max(model.predict_proba(x), axis=1) <= cfg.p1))[0]
        return index


