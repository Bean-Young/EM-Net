import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from MM.Get_field import Get_field
import h5py

def get_img_by_id(id):
    img_dir = os.path.join(imgs_dir, f"{data_prefix}{id}.{input_fex}")
    img = read_image(img_dir, ImageReadMode.RGB)

    return img

def get_field_by_id(id):
    field_dir = os.path.join(imgs_dir, f"{data_prefix}{id}.{input_fex}")
    field=Get_field(field_dir)
    field = np.expand_dims(field, axis=0)
    field = torch.from_numpy(field)
    return field

def get_msk_by_id(id):
    msk_dir = os.path.join(msks_dir, f"{data_prefix}{id}{target_postfix}.{target_fex}")
    msk = read_image(msk_dir, ImageReadMode.GRAY)
    return msk

INPUT_SIZE = 512

img_transform = transforms.Compose([
    transforms.Resize(
        size=[INPUT_SIZE, INPUT_SIZE],
        interpolation=transforms.functional.InterpolationMode.BILINEAR
    ),
])

field_transform=transforms.Compose([
    transforms.Resize(
        size=[INPUT_SIZE, INPUT_SIZE],
        interpolation=transforms.functional.InterpolationMode.BILINEAR
    ),
])


msk_transform = transforms.Compose([
    transforms.Resize(
        size=[INPUT_SIZE, INPUT_SIZE],
        interpolation=transforms.functional.InterpolationMode.NEAREST
    ),
])
def process_batch(data_ids_batch):

    for data_id in data_ids_batch:
        img = get_img_by_id(data_id)
        field = get_field_by_id(data_id)
        msk = get_msk_by_id(data_id)

        img = img_transform(img)
        field = field_transform(field)
        msk = msk_transform(msk)

        img = (img - img.min()) / (img.max() - img.min())
        field = (field - field.min()) / (field.max() - field.min())
        msk = (msk - msk.min()) / (msk.max() - msk.min())

        sample = {'image': img, 'field': field, 'label': msk}
       
        file_path = '/home/yyz/Project-Skin/data/ISIC2016/train_npz/' + data_id + '.npz'
        # Save the data as .npz file
        np.savez(file_path, **sample)
        """
        # Save the data as .npy.h5 file
        with h5py.File('/home/yyz/Project-Skin/data/ISIC2016/test_vol_h5' + data_id + '.npy.h5', 'w') as hdf_file:
            for key in sample:
                hdf_file.create_dataset(key, data=sample[key])
        """
def split_into_batches(data_ids, batch_size):
    for i in range(0, len(data_ids), batch_size):
        yield data_ids[i:i + batch_size]


def process_images_in_batches(data_ids, batch_size=24):
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_batch, split_into_batches(data_ids, batch_size)), total=len(data_ids) // batch_size))


if __name__ == "__main__":

    data_prefix = "ISIC_"
    target_postfix = "_Segmentation"
    target_fex = "png"
    input_fex = "jpg"
    data_dir = "/home/yyz/Project-Skin/data_image"
    imgs_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_Data")
    msks_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_GroundTruth")

    img_dirs = glob.glob(f"{imgs_dir}/*.{input_fex}")
    data_ids = [d.split(data_prefix)[-1].split(f".{input_fex}")[0] for d in img_dirs]

    process_images_in_batches(data_ids)