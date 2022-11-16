import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from performance import *
from utils import confidence_interval

def plot_type(type, discretized_performance, baseline_performance=None, TS_performance=None, seed=False):
    discretized = getPerformanceValue(discretized_performance, type)
    discretized_mean, discretized_err = confidence_interval(discretized)

    fig = plt.figure()
    plt.plot(cfg.quantiles, discretized_mean, linestyle='--', marker='o', color='dodgerblue', linewidth=3)
    legend=['FCCA']

    if baseline_performance is not None:
        baseline = getPerformanceValue(baseline_performance, type)
        if isinstance(baseline,np.ndarray) and baseline.ndim==2:
            baseline = baseline.flatten()
        baseline_mean, baseline_err = confidence_interval(baseline)
        plt.plot(cfg.quantiles, [baseline_mean for _ in cfg.quantiles], color='darkorange', linewidth=3)
        legend.append('baseline')
    if TS_performance is not None:
        TS = getPerformanceValue(TS_performance, type)
        if isinstance(TS,np.ndarray) and TS.ndim==2:
            TS = TS.flatten()
        TS_mean, TS_err = confidence_interval(TS)
        plt.plot(cfg.quantiles, [TS_mean for _ in cfg.quantiles], color='forestgreen', linewidth=3)
        legend.append('Target')

    plt.fill_between(cfg.quantiles, discretized_mean-discretized_err, discretized_mean+discretized_err, color='dodgerblue', alpha=0.5)
    if baseline_performance is not None:
        plt.fill_between(cfg.quantiles, baseline_mean-baseline_err, baseline_mean+baseline_err, color='darkorange', alpha=0.5)
    if TS_performance is not None:
        plt.fill_between(cfg.quantiles, TS_mean-TS_err, TS_mean+TS_err, color='forestgreen', alpha=0.5)

    plt.xlabel("Q")
    plt.ylabel(type)
    plt.legend(legend)
    title = type
    plt.title(title)
    plt.savefig(f"{cfg.getPlotDirectory(seed)}/{title}.png")
    plt.close(fig)


def plot_compression_duplicates(discretized_performance, seed=False):
    compression = getPerformanceValue(discretized_performance, 'compression')
    compression_mean, compression_err = confidence_interval(compression)

    duplicates = getPerformanceValue(discretized_performance, 'duplicates')
    duplicates_mean, duplicates_err = confidence_interval(duplicates)

    color1 = 'firebrick'
    fig,ax1 = plt.subplots()
    ax1.plot(cfg.quantiles, compression_mean, linestyle='--', marker='o', color=color1, linewidth=3)
    ax1.fill_between(cfg.quantiles, compression_mean - compression_err, compression_mean + compression_err, color=color1, alpha=0.5)
    ax1.set_xlabel('Q')
    ax1.set_ylabel('Compression Rate', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    color2 = 'lightseagreen'
    ax2 = ax1.twinx()
    ax2.plot(cfg.quantiles, duplicates_mean, linestyle='--', marker='o', color=color2, linewidth=3)
    ax2.fill_between(cfg.quantiles, duplicates_mean - duplicates_err, duplicates_mean + duplicates_err, color=color2, alpha=0.5)
    ax2.set_ylabel('Duplicates Rate', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()

    plt.savefig(f"{cfg.getPlotDirectory(seed)}/compression_duplicates.png")
    plt.close(fig)


def plot_performance(TS_performance, baseline_performance, discretized_performance, seed=False):
    plot_type('accuracy', discretized_performance, baseline_performance, TS_performance, seed=seed)
    plot_type('n_features', discretized_performance, baseline_performance, seed=seed)
    plot_compression_duplicates(discretized_performance, seed=seed)


def plot_thresholds_seed(dataset, Thresholds):
    Thresholds = pd.DataFrame.copy(Thresholds)
    height = len(np.unique(Thresholds['feature'])) * 0.4
    if len(np.unique(Thresholds['feature'])) >= 100:
        height = len(np.unique(Thresholds['feature'])) * 0.1
    figsize = (10, height)

    fig = plt.figure(figsize=figsize)

    Thresholds['threshold'] = np.floor(Thresholds['threshold'].astype(float)*10)/10
    Thresholds = Thresholds.groupby(['feature', 'threshold']).sum().reset_index()

    ThresholdsFrequency = pd.DataFrame(index=np.arange(len(dataset.features_name)), columns=np.round(np.arange(0, 1, 0.1),decimals=2), data=0)
    Thresholds = Thresholds.reset_index().pivot(index='feature', columns='threshold', values='count')
    Thresholds = Thresholds.replace(np.nan, 0)
    ThresholdsFrequency.update(Thresholds)
    sns.heatmap(ThresholdsFrequency, cmap="YlOrBr")
    plt.ylabel('Features')
    plt.xlabel('Thresholds')
    plt.title('Frequency of extracted Thresholds')
    plt.yticks(np.arange(len(dataset.features_name))+.5, dataset.features_name, rotation=0)
    if len(np.unique(ThresholdsFrequency.index))>100:
        plt.yticks([])
    plt.savefig(f"{cfg.getPlotDirectory(seed=True)}/thresholds.png")
    plt.close(fig)


def plot_counterfactual_times(TS):
    try:
        x0 = pd.read_csv(cfg.get_x_Filename('x0')).to_numpy()[:,:-1]
        proba = np.max(TS.predict_proba(x0),axis=1)
        times = pd.read_csv(cfg.get_counterfactual_log(),header=None).to_numpy().flatten()

        figure = plt.figure()
        plt.scatter(proba, times)
        plt.xlabel('classification probability')
        plt.ylabel('time (s)')
        plt.title('Time for computing the Counterfactual Explanation')
        plt.savefig(f'{cfg.getPlotDirectory(seed=True)}/counterfactual_times.png')
        plt.close(figure)
    except:
        print("Counterfactual Times are not saved")


def get_best_q(accuracy):

    if cfg.big_dataset == False:
        max_accuracy = accuracy==np.max(accuracy,axis=0)
    else:
        max_accuracy = np.abs(accuracy-np.max(accuracy,axis=0))<=0.015

    q_index = np.zeros((accuracy.shape[1],)).astype(int)
    q = np.zeros((accuracy.shape[1],))
    for k in range(accuracy.shape[1]):
        q_index[k] = np.max(np.where(max_accuracy[:,k]))
        q[k] = cfg.quantiles[q_index[k]]

    return q_index, q

def plot_features(dataset, baseline_performance, discretized_performance, Thresholds):
    baseline_features = np.zeros((len(dataset.features_name),))
    keys = list(baseline_performance.keys())

    for key in keys:
        for f in baseline_performance[key].features_used:
            baseline_features[f]+=1

    discretized_accuracy = getPerformanceValue(discretized_performance, 'accuracy')
    _, q = get_best_q(discretized_accuracy)
    print(f"best q {q}")

    discretized_features = np.zeros((len(dataset.features_name),))
    keys = list(discretized_performance.keys())
    for k in range(discretized_accuracy.shape[1]):
        for f in discretized_performance[keys[k]][q[k]].features_used:
            discretized_features[f]+=1

    Thresholds['threshold'] = np.floor(Thresholds['threshold'].astype(float)*10)/10
    Thresholds = Thresholds.groupby(['feature', 'threshold'])['count'].mean()

    ThresholdsFrequency = pd.DataFrame(index=np.arange(len(dataset.features_name)), columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=0)
    Thresholds = Thresholds.reset_index().pivot(index='feature', columns='threshold', values='count')
    Thresholds = Thresholds.replace(np.nan, 0)
    ThresholdsFrequency.update(Thresholds)
    ThresholdsFrequency = ThresholdsFrequency/np.max(ThresholdsFrequency.to_numpy())

    if len(dataset.features_name)>=100:
        height = 8
    else:
        height = 4
    figsize = (10, height)

    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1.3]})
    fig.tight_layout()

    axes[0].barh(np.arange(len(dataset.features_name))+0.5, baseline_features, align='center', color='darkorange', zorder=10)
    axes[0].set_title('Baseline')
    axes[1].barh(np.arange(len(dataset.features_name))+0.5, discretized_features, align='center', color='dodgerblue', zorder=10)
    axes[1].set_title('Best Discretized')
    axes[0].invert_xaxis()
    axes[0].invert_yaxis()

    sns.heatmap(ThresholdsFrequency, cmap="YlOrBr", ax=axes[2])
    axes[2].set_title('Thresholds')

    if len(dataset.features_name) < 30:
        axes[0].set(yticks=np.arange(len(dataset.features_name))+0.5, yticklabels=dataset.features_name)
        axes[0].yaxis.tick_left()
    else:
        axes[0].set_yticks([])

    axes[0].set_ylabel('Features')
    axes[0].set_xlabel('Frequency')
    axes[1].set_xlabel('Frequency')

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    plt.savefig(f'{cfg.getPlotDirectory()}/features.png')
    plt.close(fig)


def plot_datasets_results(TS_performance, baseline_performance, discretized_performance):
    datasets = list(TS_performance.keys())
    TS_accuracy = []
    baseline_accuracy = []
    discretized_accuracy = []
    for dataset in datasets:
        TS_accuracy.append(np.mean(getPerformanceValue(TS_performance[dataset],'accuracy')))
        baseline_accuracy.append(np.mean(getPerformanceValue(baseline_performance[dataset],'accuracy')))
        d_accuracy = getPerformanceValue(discretized_performance[dataset],'accuracy')
        q_index, _ = get_best_q(d_accuracy)
        mask = np.zeros((d_accuracy.shape))
        mask[q_index,np.arange(q_index.size)]=True
        discretized_accuracy.append(np.mean(np.sum(d_accuracy*mask,axis=0)))

    fig = plt.figure()

    plt.bar(np.arange(len(datasets)) - 0.2, TS_accuracy, width=0.2, color='forestgreen', align='center')
    plt.bar(np.arange(len(datasets)), baseline_accuracy, width=0.2, color='darkorange', align='center')
    plt.bar(np.arange(len(datasets)) + 0.2, discretized_accuracy, width=0.2, color='dodgerblue', align='center')
    plt.xticks(np.arange(len(datasets)), datasets)
    plt.legend(['Target','baseline','FCCA'])
    plt.ylim([0.66,0.95])
    plt.ylabel('Accuracy')
    plt.xlabel('Dataset')
    plt.title('Average Accuracy on all datasets')

    plt.savefig(f'{cfg.results_base_dir}/average_accuracy.png')
    plt.close(fig)
