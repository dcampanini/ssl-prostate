#%%
import numpy as np
import json
import math
import random
def subset(k):
    idx = indices[:k]
    return {
        "pat_ids": [pat_ids[i] for i in idx],
        "study_ids": [study_ids[i] for i in idx],
        "image_paths": [image_paths[i] for i in idx],
        "label_paths": [labels[i] for i in idx],
        "case_label": [case_label[i] for i in idx],
        "ratio_csPCa_bg": [ratio[i] for i in idx],
    }


save = False
train_file = '/home/dcampanini/datasets/ssl_prostate_data/overviews/UNet/overviews/Task2201_picai/PI-CAI_train-fold-0.json'

with open(train_file, "r", encoding="utf-8") as f:
    df_train = json.load(f)


pat_ids = df_train['pat_ids']
study_ids = df_train['study_ids']
image_paths = df_train['image_paths']
labels = df_train['label_paths']
case_label = df_train['case_label']
ratio = df_train['ratio_csPCa_bg']

n = len(pat_ids)

n_25 = math.floor(0.25 * n)
n_50 = math.floor(0.50 * n)
n_75 = math.floor(0.75 * n)

# %%

indices = list(range(n))
random.seed(42)  # reproducibility
random.shuffle(indices)
splits = {
    "25": subset(n_25),
    "50": subset(n_50),
    "75": subset(n_75),
}

# %%
print('original total data = ', n)
output= "/home/dcampanini/datasets/ssl_prostate_data/overviews/UNet/overviews"
if save:
    for k, split_data in splits.items():
        print('spit data', k,'total data =',len(split_data['pat_ids']))
        output_path = f"{output}/Task2201_picai_{str(k)}/PI-CAI_train-fold-0.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=4)

# %%
