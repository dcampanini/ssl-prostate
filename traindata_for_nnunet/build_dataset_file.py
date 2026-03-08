#%%
import shutil
import numpy as np
import json
import os 

output_path = "/workspace1/project_jxfdv/ssl_prostate_data/nnUNet_raw_data"
dataset = "uc"

for f in [0,1,2,3,4]:
    if dataset == "p158":
        # task
        # target_dir
        train_file = f'/workspace1/project_jxfdv/ssl_prostate_data/overviews/UNet/overviews/Task2401_p158_prostate/PI-CAI_train-fold-{f}.json'
        name = "p158_prostate_nnunet"
    elif dataset == "uc":
        name = "uc_prostate_nnunet"
        task = f"Task2303{f}_uc_prostate_nnunet_fold_{f}"
        target_dir = f"{output_path}/Task2303{f}_uc_prostate_nnunet_fold_{f}"
        train_file = f'/workspace1/project_jxfdv/ssl_prostate_data/overviews/UNet/overviews/Task2301_uc_prostate/PI-CAI_train-fold-{f}.json'

    os.makedirs(target_dir, exist_ok=True)
    with open(train_file, "r", encoding="utf-8") as f:
        df_train = json.load(f)
    
    ids = df_train['pat_ids']
    labels = df_train['label_paths']
    labels.sort()

    training = []

    for i in range(len(labels)):
        file_name = labels[i].split("/")[-1]
        print(file_name)
        training.append({
            "image": f"./imagesTr/{file_name}",
            "label": f"./labelsTr/{file_name}"
        })
        #break
    
    print(f"Training images and labels saved for fold {f}")
    print("Total studies saved:", i+1)

    #%
    final_dict = {
            "task": task,
            "description": "bpMRI scans to train nnUNet baseline",
            "tensorImageSize": "4D",
            "reference": "",
            "licence": "",
            "release": "1.0",
            "modality": {
                "0": "T2W",
                "1": "CT",
                "2": "HBV"
            },
            "labels": {
                "0": "background",
                "1": "lesion"
            },
            "name": name,
            "numTraining": len(training),
            "training": training,
            "numTest": 0,
            "test": []
        }
    
    with open(f"{target_dir}/dataset.json", "w") as f:
        json.dump(final_dict, f, indent=4)


# %%
