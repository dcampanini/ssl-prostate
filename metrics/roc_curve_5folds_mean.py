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
#json_path = '/home/dcampanini/ssl-prostate/metrics/p158'
json_path = '/home/dcampanini/ssl-prostate/metrics/uc'

# Define all three model groups
# model_groups = {
#     'UNETR-FULL-DAE': [
#         'unetr_full_dae_p158f0',
#         'unetr_full_dae_p158f1',
#         'unetr_full_dae_p158f2',
#         'unetr_full_dae_p158f3',
#         'unetr_full_dae_p158f4'
#     ],
#     'UNET Scratch': [
#         'unet_scratch_p158_f0',
#         'unet_scratch_p158_f1',
#         'unet_scratch_p158_f2', 
#         'unet_scratch_p158_f3',
#         'unet_scratch_p158_f4'
#     ],
#     'UNET-FULL-SCLR-MAE': [
#         'unet_(1,2,3)_SCLR-MAE_p158_f0',
#         'unet_(1,2,3)_SCLR-MAE_p158_f1',
#         'unet_(1,2,3)_SCLR-MAE_p158_f2',
#         'unet_(1,2,3)_SCLR-MAE_p158_f3',
#         'unet_(1,2,3)_SCLR-MAE_p158_f4'
#     ],
# }


model_groups = {
    'UNET-FULL-BYOL-MAE': [
        'unet_full_byol_mae_uc_f0',
        'unet_full_byol_mae_uc_f1',
        'unet_full_byol_mae_uc_f2',
        'unet_full_byol_mae_uc_f3',
        'unet_full_byol_mae_uc_f4'
    ],
    'UNET Scratch': [
        'unet_scratch_uc_f0',
        'unet_scratch_uc_f1',
        'unet_scratch_uc_f2', 
        'unet_scratch_uc_f3',
        'unet_scratch_uc_f4'
    ],
    'UNET-(1-3)-MAE': [
        'unet_(1-3)_mae_uc_f0',
        'unet_(1-3)_mae_uc_f1',
        'unet_(1-3)_mae_uc_f2',
        'unet_(1-3)_mae_uc_f3',
        'unet_(1-3)_mae_uc_f4'
    ],
}

# Define colors for each model
colors = ['blue', 'red', 'green']

mean_fpr = np.linspace(0, 1, 100)

# Create the plot
plt.figure(figsize=(8, 6))

# Process each model group
for (model_name, models), color in zip(model_groups.items(), colors):
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
        aucs.append(np.round(roc_auc, 3))
    
    tprs = np.array(tprs)
    
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)
    
    mean_tpr[-1] = 1.0
    
    # Plot mean ROC curve
    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        lw=2,
        label=f"{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
    )
    
    # Plot standard deviation band
    plt.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color=color,
        alpha=0.2
    )

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color='gray')

plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.title("Mean ROC curves for best models - ChiPCa", fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)

# Save figure
plt.savefig(f"{save_fig_path}/roc_curve_uc.pdf", format="pdf", bbox_inches="tight")
plt.show()
# %%
