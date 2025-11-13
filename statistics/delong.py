import numpy as np
from sklearn.metrics import roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
import json
from scipy.stats import norm

def delong_roc_variance(ground_truth, predictions):
    """
    Compute the variance of the AUC using DeLong's method.
    Arguments:
    ground_truth -- true binary labels
    predictions -- predicted scores
    Returns:
    auc -- area under the ROC curve
    auc_var -- variance of the AUC
    """
    # ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
    auc = np.trapz(tpr, fpr)
    
    # DeLong variance estimation
    positives = predictions[ground_truth == 1]
    negatives = predictions[ground_truth == 0]
    
    pos_count = len(positives)
    neg_count = len(negatives)
    
    # Compute the AUC scores for positive and negative samples
    auc_scores = np.zeros((pos_count, neg_count))
    for i in range(pos_count):
        for j in range(neg_count):
            if positives[i] > negatives[j]:
                auc_scores[i, j] = 1
            elif positives[i] == negatives[j]:
                auc_scores[i, j] = 0.5
    
    # Variance calculation
    pos_var = np.var(np.mean(auc_scores, axis=1)) / pos_count
    neg_var = np.var(np.mean(auc_scores, axis=0)) / neg_count
    auc_var = pos_var + neg_var
    
    return auc, auc_var

def delong_test(y_true, y_pred_1, y_pred_2):
    """
    Realiza la prueba de DeLong para comparar dos AUROCs.
    
    :param y_true: Array con las etiquetas verdaderas (0 o 1).
    :param y_pred_1: Array con las predicciones del primer modelo.
    :param y_pred_2: Array con las predicciones del segundo modelo.
    :return: p-value de la prueba de DeLong.
    """
    auc_1, auc_var_1 = delong_roc_variance(y_true, y_pred_1)
    auc_2, auc_var_2 = delong_roc_variance(y_true, y_pred_2)
    print('AUC 1:', auc_1, 'AUC VAR 1:', auc_var_1)
    print('AUC 2:', auc_2, 'AUC VAR 2:', auc_var_2)
    
    diff = auc_1 - auc_2
    var_diff = auc_var_1 + auc_var_2
    
    z_score = diff / np.sqrt(var_diff)
    p_value = 2 * stats.norm.cdf(-abs(z_score))  # Doble cola para obtener el p-value
    
    return p_value, auc_1, auc_var_1, auc_2, auc_var_2

# Ejemplo de uso

def use_delong(path_metrics1, path_metrics2):
    f1 = open(path_metrics1)
    f2 = open(path_metrics2)
    metrics1 = json.load(f1)
    metrics2 = json.load(f2)
    f1.close()
    f2.close()
    y_true = np.array(list(metrics1['case_target'].values()))
    y_pred1 = np.array(list(metrics1['case_pred'].values()))
    y_pred2 = np.array(list(metrics2['case_pred'].values()))
    p_value, auc_1, auc_var_1, auc_2, auc_var_2 = delong_test(y_true, y_pred1, y_pred2)
    ci_lower1, ci_upper1 = auc_confidence_interval(auc_1, auc_var_1)
    ci_lower2, ci_upper2 = auc_confidence_interval(auc_2, auc_var_2)
    return p_value , ci_lower1, ci_upper1, ci_lower2, ci_upper2

def auc_confidence_interval(auc, auc_var, alpha=0.95):
    """
    Calculates the confidence interval for the AUC using the variance from DeLong.
    
    :param auc: AUROC value.
    :param auc_var: Variance of the AUROC.
    :param alpha: Confidence level.
    :return: (lower bound, upper bound) of the confidence interval.
    """
    # Z-score for the given alpha
    z = norm.ppf(1 - (1 - alpha) / 2)
    lower_bound = auc - z * np.sqrt(auc_var)
    upper_bound = auc + z * np.sqrt(auc_var)
    
    return lower_bound, upper_bound

metrics_base_path = '/mnt/researchers/denis-parra/datasets/jfacuse_workdir/metrics-uc/'
metrics1 = 'UNETR_FULL_MAE.json'
metrics2 = 'UNETR_Scratch.json'
p_value , ci_lower1, ci_upper1, ci_lower2, ci_upper2 = use_delong(metrics_base_path + metrics1, metrics_base_path + metrics2)
print(f'Prueba de DeLong p-value: {p_value}')
print(f'Intervalo de confianza del AUC: [{ci_lower1:.3f}, {ci_upper1:.3f}]')
print(f'Intervalo de confianza del AUC: [{ci_lower2:.3f}, {ci_upper2:.3f}]')