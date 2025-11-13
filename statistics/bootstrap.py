from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import numpy as np
import json
from picai_eval import Metrics

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    """
    Calcula intervalos de confianza para AUROC usando bootstrapping.
    
    :param y_true: Array con las etiquetas verdaderas (0 o 1).
    :param y_pred: Array con las predicciones del modelo.
    :param n_bootstraps: Número de muestras bootstrap.
    :param alpha: Nivel de confianza.
    :return: Intervalo de confianza para el AUROC.
    """
    bootstrapped_aucs = []
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = resample(range(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            # Skip iteration if there is only one class in the bootstrap sample
            continue
        
        auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(auc)
    
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()
    
    # Percentile method for CI
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    
    return lower_bound, upper_bound

def bootstrap_AP(path_metrics, n_bootstraps=1000, alpha=0.95):
    metrics = Metrics(path_metrics)
    #print(metrics.subject_list)
    bootstrapped_aps = []
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = resample(metrics.subject_list, replace=True)
        #print(indices)
        if len(np.unique([metrics.case_target[s] for s in indices])) < 2:
            # Skip iteration if there is only one class in the bootstrap sample
            continue
        ap = metrics.calc_AP(subject_list=indices)
        bootstrapped_aps.append(ap)
    
    sorted_scores = np.array(bootstrapped_aps)
    sorted_scores.sort()
    
    # Percentile method for CI
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    
    return lower_bound, upper_bound


def use_bootstrap(path_metrics1):
    f1 = open(path_metrics1)
    metrics1 = json.load(f1)
    f1.close()
    y_true = np.array(list(metrics1['case_target'].values()))
    y_pred1 = np.array(list(metrics1['case_pred'].values()))
    lower_bound, upper_bound = bootstrap_auc(y_true, y_pred1)
    return lower_bound, upper_bound

metrics_base_path = '/mnt/researchers/denis-parra/datasets/jfacuse_workdir/metrics-uc/'
metrics1 = 'unetr_(1,2)_dae_fulltest.json'
ci_lower, ci_upper = use_bootstrap(metrics_base_path + metrics1)
ci_lower_ap, ci_upper_ap = bootstrap_AP(metrics_base_path + metrics1)
print(f'Intervalo de confianza del AUROC: [{ci_lower:.3f}, {ci_upper:.3f}]')
print(f'Intervalo de confianza del AP: [{ci_lower_ap:.3f}, {ci_upper_ap:.3f}]')