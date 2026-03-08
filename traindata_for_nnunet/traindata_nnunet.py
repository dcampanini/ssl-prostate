#%%
import shutil
import numpy as np
import json
import os 

output_path = "/workspace1/project_jxfdv/ssl_prostate_data/nnUNet_raw_data"
dataset = "uc"
for f in [0,1,2,3,4]:
    if dataset == "p158":
        target_dir_labels = f"{output_path}/Task2403_p158_prostate_nnunet_fold_{f}/labelsTr"
        target_dir_imgs = f"{output_path}/Task2403_p158_prostate_nnunet_fold_{f}/imagesTr"
        train_file = f'/workspace1/project_jxfdv/ssl_prostate_data/overviews/UNet/overviews/Task2401_p158_prostate/PI-CAI_train-fold-{f}.json'
    elif dataset == "uc":
        target_dir_labels = f"{output_path}/Task2303{f}_uc_prostate_nnunet_fold_{f}/labelsTr"
        target_dir_imgs = f"{output_path}/Task2303{f}_uc_prostate_nnunet_fold_{f}/imagesTr"
        train_file = f'/workspace1/project_jxfdv/ssl_prostate_data/overviews/UNet/overviews/Task2301_uc_prostate/PI-CAI_train-fold-{f}.json'
    
    
    os.makedirs(target_dir_labels, exist_ok=True)
    os.makedirs(target_dir_imgs, exist_ok=True)
    with open(train_file, "r", encoding="utf-8") as f:
        df_train = json.load(f)
    
    ids = df_train['pat_ids']
    labels = df_train['label_paths']
    images = df_train['image_paths']

    for i in range(len(labels)):
        shutil.copy(labels[i], target_dir_labels)
        for img in images[i]:
            shutil.copy(img, target_dir_imgs)
        #break
    
    print(f"Training images and labels saved for fold {f}")
    print("Total studies saved:", i+1)


