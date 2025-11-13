import json
import os 
"""
Hay que crear un json con formato:
[
    {
    "train": [Brats2021_0001, ....],
    "val": [Brats2021_0002, ...]
    }
]

La idea es que en training hayan solo casos negativos,
es decir que no tengan nada en el label, y en validacion ponemos a los positivos"""

out_file = './data/brats2021/splits.json'
pos_cases_file = '/home/jfacuse/brats_pos_cases.txt'
labels_folder = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Dataset137_BraTS2021/labelsTr'
split_dict = {'train': [], 'val': []}
f = open(pos_cases_file, 'r')
pos_cases = f.readlines()
l_folder = os.listdir(labels_folder)
print(pos_cases)
for i, case in enumerate(l_folder):
    if i < len(l_folder) * 0.8:
        split_dict['train'].append(case.strip('.nii.gz'))
    else:
        split_dict['val'].append(case.strip('.nii.gz'))

f.close()
with open(out_file, 'w', encoding='utf-8') as p:
    json.dump([split_dict], p, ensure_ascii=False, indent=4)