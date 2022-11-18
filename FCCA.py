import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from performance import Performance
from config import cfg

def discretization_parameters(n_features, Thresholds):
    features_categories = []
    features_boundaries = []
    for i in range(n_features):
        thresholds = Thresholds[Thresholds['feature'] == i]['threshold'].to_list()
        if len(thresholds) > 0:
            boundaries = [0] + thresholds + [1]
            thresholds = [-100] + thresholds + [100]
            features_boundaries.append(boundaries)
        else:
            features_boundaries.append([])
        features_categories.append(thresholds)
    return features_categories, features_boundaries


def discretize(x, features_categories):
    x = np.copy(x)
    for i in range(len(features_categories)):
        if len(features_categories[i]) > 0:
            x[:, i] = pd.cut(x[:, i], features_categories[i], labels=np.around(np.arange(len(features_categories[i]) - 1) / (len(features_categories[i]) - 2), 4))
    return x


def selectFeatures(x, features_categories):
    features_mask = np.array([True if len(i) > 0 else False for i in features_categories])
    x = x[:, features_mask]
    return x


def encode(x, features_categories):
    x = discretize(x, features_categories)
    x = selectFeatures(x, features_categories)
    return x


def compress(x_init, y_init, features_categories):
    x = encode(x_init, features_categories)
    y = y_init
    u = np.sort(np.unique(x, axis=0, return_index=True)[1])
    count = 0
    j = 0
    y_collapsed = []
    for i in range(len(x)):
        if i in u:
            si = np.where((x == x[i, :]).all(axis=1))[0]
            yy, cc = np.unique(y[si], return_counts=True)
            if len(yy) > 1:
                count += np.min(cc)
            y_collapsed.append(yy[np.argmax(cc)])
            j += 1
    inconsistency_rate = count/len(x)

    x_compressed = x[u, :]
    r = len(x_compressed) / len(x_init)
    compression_rate = 1 - r

    return x, y, inconsistency_rate, compression_rate


def computeThresholds(x0, xCE, eps):
    diff = x0 - xCE

    log = open(cfg.get_seed_log_Filename(),'w')
    features_changed = np.sum(abs(diff) > 1.e-4, axis=1)
    print(f"Average number of features changed by the counterfactuals {np.mean(features_changed)}", file=log)
    print(f"Min number of features changed by the counterfactuals {np.min(features_changed)}", file=log)
    print(f"Median number of features changed by the counterfactuals {np.quantile(features_changed, 0.5)}",file=log)
    print(f"Max number of features changed by the counterfactuals {np.max(features_changed)}", file=log)

    Thresholds = pd.DataFrame(columns=['feature', 'count', 'threshold'])
    for i in range(diff.shape[1]):
        mask = abs(diff[:, i]) > 1.e-4
        c = xCE[:, i][mask] + 1.1 * eps[i] * np.sign(diff[:, i][mask])
        t = pd.DataFrame({'threshold': c, 'round': np.round(c, decimals=cfg.n_decimals)})
        thresholds = t.groupby('round').median()
        thresholds['count'] = t.groupby('round').count()
        thresholds['feature'] = i
        Thresholds = pd.concat((Thresholds, thresholds.reset_index().drop('round', axis=1)))
    return Thresholds


def FCCA(dataset, TS):
    log = open(cfg.get_log_Filename(), 'a')
    seed_log = open(cfg.get_seed_log_Filename(), 'a')

    # Computing initial points
    index_0 = dataset.getRelevant(dataset.x_tr, dataset.y_tr, 0, TS.model)
    index_1 = dataset.getRelevant(dataset.x_tr, dataset.y_tr, 1, TS.model)
    x0 = np.concatenate((dataset.x_tr[index_0, :], dataset.x_tr[index_1, :]), axis=0)
    n_init = len(x0)

    if cfg.p2 < 1:
        centers = KMeans(n_clusters=int(len(x0) * cfg.p2), n_init=100, random_state=cfg.seed).fit(x0).cluster_centers_
        x0 = x0[np.argsort(distance_matrix(centers, x0))[:, 0], :]
    y0 = TS.predict(x0)

    print(f"{cfg.seed}, {TS.model.n_estimators}, {TS.model.max_depth}, {accuracy_score(dataset.y_tr, TS.model.predict(dataset.x_tr))}, {accuracy_score(dataset.y_ts, TS.model.predict(dataset.x_ts))}, {n_init}, {len(x0)}, {np.unique(y0,return_counts=True)[1][0]}, {np.unique(y0,return_counts=True)[1][1]}", file=log)
    log.flush()
    # Computing Counterfactual Explanations
    yCE_required = [cfg.counterfactual_labels[i] for i in y0]
    xCE, yCE = TS.getCEs(x0, yCE_required)

    # Computing thresholds
    Thresholds = computeThresholds(x0, xCE, dataset.eps)

    # Baseline model
    baseline_model = DecisionTreeClassifier(max_depth=cfg.d)
    baseline_model.fit(dataset.x_tr, dataset.y_tr)
    baseline_performance = Performance(baseline_model)
    baseline_performance.evaluate(dataset.x_ts, dataset.y_ts)

    discretized_performance = {}
    for quantile_level in cfg.quantiles:
        FrequentThresholds = Thresholds[Thresholds['count'] >= Thresholds['count'].quantile(quantile_level)]

        print(f"Q={quantile_level} - Frequency value: {Thresholds['count'].quantile(quantile_level)}", file=seed_log)
        print(f"{len(FrequentThresholds)} thresholds on {len(np.unique(FrequentThresholds['feature']))} different features",file=seed_log)
        print(FrequentThresholds, file=seed_log)

        features_categories, features_boundaries = discretization_parameters(dataset.x_tr.shape[1], FrequentThresholds)

        x_tr_discretized, y_tr_discretized, inconsistency_rate_tr, compression_rate_tr = compress(dataset.x_tr, dataset.y_tr, features_categories)
        x_ts_discretized, y_ts_discretized, inconsistency_rate_ts, compression_rate_ts = compress(dataset.x_ts, dataset.y_ts, features_categories)

        discretized_model = DecisionTreeClassifier(max_depth=cfg.d)
        discretized_model.fit(x_tr_discretized, y_tr_discretized)
        discretized_performance[quantile_level] = Performance(discretized_model, compression_rate_ts, inconsistency_rate_ts)
        discretized_performance[quantile_level].evaluate(x_ts_discretized, y_ts_discretized)
        discretized_performance[quantile_level].features_used = np.array([i for i in range(len(features_categories)) if features_categories[i]!=[]])[discretized_performance[quantile_level].features_used]
    return Thresholds, baseline_performance, discretized_performance




