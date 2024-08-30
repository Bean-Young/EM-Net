import os
from show import from_nii_to_png
from concect import get_output_image
from loss import get_loss

path='/home/yyz/Project-Skin/predictions/TU_ISIC2016224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs12_224'
inpath='/home/yyz/Project-Skin/data/ISIC2016/test_vol_h5'
mspath='/home/yyz/Project-Skin/data/ISIC2016/test_vol_h5'
outpath='/home/yyz/Project-Skin/output_image'
l=[]
data_ids = []
loss = {'dice_coefficient': 0,
       'sensitivity': 0,
       'specificity': 0,
       'accuracy': 0,
       'precision': 0,
       'f1_score': 0,
       'mean_iou': 0}

for filename in os.listdir(path):
    if filename.endswith('_pred.npz'):
        data_id = filename.split('_pred.npz')[0]
        if len(data_id) == 7:
            if data_id not in l:
                data_ids.append(data_id)

for data_id in data_ids:
    pre_path=path+'/'+data_id+'_pred.npz'
    input_path=inpath+'/'+data_id+'.npy.h5'
    mask_path=mspath+'/'+data_id+'.npy.h5'
    preim_path=outpath+'/ISIC2016_mask/'+data_id+'.png'
    outim_path=outpath+'/ISIC2016_concat/'+data_id+'.png'
    from_nii_to_png(pre_path,preim_path)
    get_output_image(input_path,mask_path,preim_path,outim_path)
    #print(data_id)
    iou=[]
    dice=[]
    for key, value in get_loss(mask_path,preim_path).items():
        if (key=="mean_iou"):
            iou.append(round(value,4))
        if (key=='dice_coefficient'):
            dice.append(value)
        if (key=="mean_iou") and (value<0.5):
            print(data_id)
        loss[key] += value


loss = {key: round(value / len(data_ids), 4) for key, value in loss.items()}
import pandas as pd

df = pd.DataFrame(list(loss.items()), columns=['Key', 'Value'])

excel_path = outpath+'/loss/ISIC_loss.xlsx'
df.to_excel(excel_path, index=False)
