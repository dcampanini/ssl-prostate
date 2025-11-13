"""
Este archivo sirve para agarrar el dataset.json y el splits.json y crear un unico json tipo split1.json 
que tenga la información de el determinado split listo para usar con las funciones del framework SelfMedMAE
"""
import json

dataset_json_path = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/dataset.json'
splits_path = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/splits.json'
write_path = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/'

d = open(dataset_json_path)
s = open(splits_path)

dataset = json.load(d)
splits = json.load(s)
#print('SPLITS', splits)
#print('DATASET', dataset)

for n, split in enumerate(splits):
    print(f'Creando split {n}')
    new_split = dataset.copy()
    new_training = list(filter(lambda x: x['image'].split('/')[2].split('.')[0] in split['train'], dataset['training']))
    new_val = list(filter(lambda x: x['image'].split('/')[2].split('.')[0] in split['val'], dataset['training']))
    new_split['training'] = new_training
    new_split['validation'] = new_val
    with open(write_path + f"split{n}.json", 'w') as outfile:
        json.dump(new_split, outfile)
    
    #print('SPLIT', split)
    

d.close()
s.close()