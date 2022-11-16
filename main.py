import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from target import TargetSystem
from plot import *
from FCCA import FCCA
from config import cfg

if __name__ == "__main__":
    datasets = ['boston','arrhythmia','ionosphere','magic','particle','vehicle']

    TS_performance = {}
    baseline_performance = {}
    discretized_performance = {}
    for name in datasets:
        cfg.set_name(name)
        TS_performance[name] = {}
        baseline_performance[name] = {}
        discretized_performance[name] = {}

        Thresholds = pd.DataFrame(columns=['seed','feature','threshold','count'])

        log = open(cfg.get_log_Filename(),'w')
        print("seed, n_estimators, max_depth, accuracy_tr, accuracy_ts, len_x0, n_counterfactuals, n_counterfactuals_0, n_counterfactuals_1", file=log)
        log.flush()

        for seed in range(cfg.min_seed, cfg.max_seed):
            print(f"seed {seed}")
            cfg.seed = seed
            np.random.seed(seed)
            dataset = Dataset(test_size=cfg.test_split)
            TS = TargetSystem(dataset)
            TS_performance[name][seed] = Performance(TS.model)
            TS_performance[name][seed].evaluate(dataset.x_ts, dataset.y_ts)

            Thresholds_seed, baseline_performance[name][seed], discretized_performance[name][seed] = FCCA(dataset, TS)
            plot_performance(TS_performance[name][seed], baseline_performance[name][seed], discretized_performance[name][seed], seed=True)
            plot_thresholds_seed(dataset, Thresholds_seed)
            Thresholds_seed['seed']=seed
            Thresholds = Thresholds.append(Thresholds_seed)
            plot_counterfactual_times(TS)

        plot_performance(TS_performance[name], baseline_performance[name], discretized_performance[name])
        plot_features(dataset, baseline_performance[name], discretized_performance[name], Thresholds)

        results = pd.DataFrame(columns=cfg.quantiles, data=getPerformanceValue(discretized_performance[name],'accuracy').transpose())
        results['seed'] = range(cfg.min_seed, cfg.max_seed)
        results['Target'] = getPerformanceValue(TS_performance[name],'accuracy')
        results['Baseline'] = getPerformanceValue(baseline_performance[name],'accuracy')
        results = results[['seed','Target','Baseline']+cfg.quantiles]
        results.to_csv(f'{cfg.getResultsDir()}/results.txt',float_format='%.3f',index=False)

    plot_datasets_results(TS_performance, baseline_performance, discretized_performance)

