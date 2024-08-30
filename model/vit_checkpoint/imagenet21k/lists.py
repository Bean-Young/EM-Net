import os

folder_path_1 = '/home/yyz/Project-Skin/data/ISIC2016/train_npz'
output_file_1 = '/home/yyz/Project-Skin/lists/lists_ISIC2016/train.txt'
folder_path_2='/home/yyz/Project-Skin/data/ISIC2016/test_vol_h5'
output_file_2='/home/yyz/Project-Skin/lists/lists_ISIC2016/test_vol.txt'
output_file_3='/home/yyz/Project-Skin/lists/lists_ISIC2016/all.lst'

with open(output_file_1, 'w') as f:
    for filename in os.listdir(folder_path_1):
        if filename.endswith('.npz'):
            name_without_extension = os.path.splitext(filename)[0]
            f.write(name_without_extension + '\n')

with open(output_file_2, 'w') as f:
    for filename in os.listdir(folder_path_2):
        if filename.endswith('.npy.h5'):
            name_without_extension = os.path.splitext(os.path.splitext(filename)[0])[0]
            f.write(name_without_extension + '\n')

with open(output_file_3, 'w') as f:
    for filename in os.listdir(folder_path_2):
        if filename.endswith('.npy.h5'):
            name_without_extension = filename
            f.write(name_without_extension + '\n')