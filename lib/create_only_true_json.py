"""
Este archivo toma uno de los overviews de training y validacion
y crea uno similar pero que solo tiene casos negativos y/o positivos
"""

import json
import os
label_to_keep = 0.0 #Cambiar a 1.0 si se quiere mantener casos CON cancer
overviews_dir = '/mnt/workspace/jfacuse/prostate/workdir/results/UNet/overviews/Task2208_picai_prostate158'
new_overviews_dir = '/mnt/workspace/jfacuse/prostate/workdir/results/UNet/overviews/Task2210_picai_prostate158_only_negative/'
if not os.path.isdir(new_overviews_dir):
    os.mkdir(new_overviews_dir)
for overview in os.listdir(overviews_dir):
    path = os.path.join(overviews_dir, overview)
    print(path)
    o = open(path)
    overview_json = json.load(o)
    print(len(overview_json['case_label']),
           len(overview_json['image_paths']),
             len(overview_json['label_paths']),
               len(overview_json['ratio_csPCa_bg']), len(overview_json['pat_ids']), len(overview_json['study_ids']))
    new_json = {}
    cancer_indices = [i for i, x in enumerate(overview_json['case_label']) if x == label_to_keep]
    new_json['case_label'] = [overview_json['case_label'][x] for x in cancer_indices]
    new_json['image_paths'] = [overview_json['image_paths'][x] for x in cancer_indices]
    new_json['label_paths'] = [overview_json['label_paths'][x] for x in cancer_indices]
    new_json['ratio_csPCa_bg'] = [overview_json['ratio_csPCa_bg'][x] for x in cancer_indices]
    new_json['pat_ids'] = [overview_json['pat_ids'][x] for x in cancer_indices]
    new_json['study_ids'] = [overview_json['study_ids'][x] for x in cancer_indices]
    with open(new_overviews_dir + overview, 'w') as outfile:
        json.dump(new_json, outfile)
    
    o.close()
