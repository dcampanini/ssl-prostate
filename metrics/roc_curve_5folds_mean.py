#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import json
#%%
save_fig_path = '/home/dcampanini/ssl-prostate/metrics/figures'
json_path = '/home/dcampanini/ssl-prostate/metrics/unetr/stage3_p158'
model = 'UNETR FULL DAE' # use paper name
models = [
    'unetr_full_dae_p158f0',
    'unetr_full_dae_p158f1',
    'unetr_full_dae_p158f2',
    'unetr_full_dae_p158f3',
    'unetr_full_dae_p158f4'
]

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

for m in models:
    print(m)
    # read a json file
    with open(f'{json_path}/{m}/roc_curve_data.json', 'r') as f:
        data = json.load(f)
    # access fpr and tpr
    fpr = data['FPR']
    tpr = data['TPR']
    roc_auc = data['AUROC']
    
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0

    tprs.append(tpr_interp)
    aucs.append(roc_auc)

tprs = np.array(tprs)

mean_tpr = tprs.mean(axis=0)
std_tpr = tprs.std(axis=0)

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

mean_tpr[-1] = 1.0

plt.plot(
    mean_fpr,
    mean_tpr,
    lw=2,
    label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
)

plt.fill_between(
    mean_fpr,
    mean_tpr - std_tpr,
    mean_tpr + std_tpr,
    alpha=0.2,
    #label="±1 std"
)

plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Mean ROC Curve model {model}")
plt.legend()
# save figure
plt.savefig(f"{save_fig_path}/roc_curve_{model}.pdf", format="pdf", bbox_inches="tight")
plt.show()  