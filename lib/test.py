from monai.transforms import LoadImage, SaveImage
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
test_img = '/mnt/data/MSD-Brats/Task01_BrainTumour/imagesTr/BRATS_155.nii.gz'
data = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(test_img)
print(f"image data shape: {data.shape}")

test_img_prostate = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/imagesTr/10802_1000818.nii.gz'
test_img_prostate2 = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/imagesTr/10802_1000818_0000.nii.gz'
data2 = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(test_img_prostate)
data3 = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)('/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/imagesTr/10802_1000818_0000.nii.gz')
print(f"image data shape: {data2.shape}")
print('Son iguales:', data2[0] == data3)
data4 = np.expand_dims(sitk.GetArrayFromImage(
                sitk.ReadImage(test_img_prostate2)
            ).astype(np.float32), axis=(0, 1)
        )
print('d4 shape', data4.shape)

folder = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/imagesTr'
"""
for path in filter(lambda x: '0000.nii.gz' in x,os.listdir(folder)):
    # Cargamos todas las imagenes para juntarlas en una
    adc_path = path.replace('0000.', '0001.')
    dwi_path = path.replace('0000.', '0002.')
    t2 = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(os.path.join(folder,path))
    adc = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(os.path.join(folder,adc_path))
    dwi = LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(os.path.join(folder,dwi_path))
    full = np.concatenate([t2,adc,dwi], axis=0)
    print(full.shape)
    name = path[:-12]
    nib_img = nib.Nifti1Image(full, np.eye(4))
    nib.save(nib_img, os.path.join(folder,f"{name}.nii.gz"))
    print(path, full.shape)
"""    
    


for path in filter(lambda x: len(x) < 22, os.listdir(folder)):
    img = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(os.path.join(folder,path))
    print(path, img.shape)
    
