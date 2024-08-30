import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def add_black_border(img, border_size=2):
    # Create a new image with a larger size initialized as black
    if img.shape[0] == 1:  # Single-channel image
        new_img = np.zeros((1, img.shape[1]+2*border_size, img.shape[2]+2*border_size), dtype=np.float32)
    else:  # RGB image
        new_img = np.zeros((3, img.shape[1]+2*border_size, img.shape[2]+2*border_size), dtype=np.float32)
    new_img[:, border_size:img.shape[1]+border_size, border_size:border_size+img.shape[2]] = img
    new_img = torch.from_numpy(new_img.astype(np.float32))
    return new_img

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.rotation_transform = transforms.RandomRotation(degrees=(-30, 30))
        self.flip_transform = transforms.RandomHorizontalFlip()
        self.resize_transform = transforms.Resize(self.output_size)
        self.brightness_transform = transforms.ColorJitter(brightness=(0.5, 1.5))
        self.crop_transform = transforms.RandomResizedCrop(self.output_size)

    def transform_sample(self, image, field, label, transform):
        seed = np.random.randint(0, 2032)
        torch.manual_seed(seed)
        transformed_image = transform(image)
        torch.manual_seed(seed)
        transformed_field = transform(field)
        torch.manual_seed(seed)
        transformed_label = transform(label)
        return transformed_image, transformed_field, transformed_label

    def transform_sample_4(self, image, field, label, transform):
        img_transformed = transform(image)
        field_transformed = field
        label_transformed = label
        return img_transformed, field_transformed, label_transformed

    def __call__(self, sample):
        image, field, label = sample['image'], sample['field'], sample['label']
        samples = []  # To store augmented sample set
        image = torch.from_numpy(image.astype(np.float32))
        field = torch.from_numpy(field.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image, field, label = self.transform_sample(image, field, label, self.resize_transform)
        # Original sample
        original_image, original_field, original_label = image, field, label
        original_image, original_field, original_label = add_black_border(original_image), add_black_border(original_field), add_black_border(original_label)
        original_image, original_field, original_label = self.transform_sample(original_image, original_field, original_label, self.resize_transform)
        samples.append({
            'image': original_image,
            'field': original_field,
            'label': original_label.long()
        })

        # Random flip
        flipped_image, flipped_field, flipped_label = self.transform_sample(image, field, label, self.flip_transform)
        flipped_image, flipped_field, flipped_label = add_black_border(flipped_image), add_black_border(flipped_field), add_black_border(flipped_label)
        flipped_image, flipped_field, flipped_label = self.transform_sample(flipped_image, flipped_field, flipped_label, self.resize_transform)
        samples.append({
            'image': flipped_image,
            'field': flipped_field,
            'label': flipped_label.long()
        })

        # Random crop
        cropped_image, cropped_field, cropped_label = self.transform_sample(image, field, label, self.crop_transform)
        cropped_image, cropped_field, cropped_label = add_black_border(cropped_image), add_black_border(cropped_field), add_black_border(cropped_label)
        cropped_image, cropped_field, cropped_label = self.transform_sample(cropped_image, cropped_field, cropped_label, self.resize_transform)
        samples.append({
            'image': cropped_image,
            'field': cropped_field,
            'label': cropped_label.long()
        })

        return samples

def rgb_to_single_channel_with_color_diff(rgb_array):
    r_weight = 0.1890
    g_weight = 0.3370
    b_weight = 0.4740
    single_channel_array = np.dot(rgb_array[..., :3], [r_weight, g_weight, b_weight])
    single_channel_array = np.clip(single_channel_array, 0, 255).astype('uint8')
    return single_channel_array

class Skin_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, field, label = data['image'], data['field'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, field, label = data['image'][:], data['field'][:], data['label'][:]

        sample = {'image': image, 'field': field, 'label': label}
        additional = sample['image'][()] * 255
        additional = additional.astype('uint8')
        additional = additional.transpose((1, 2, 0))
        blue_emphasized_array = rgb_to_single_channel_with_color_diff(additional)
        additional_img_gray = sample['field'][()].squeeze(0) * 255
        added_images_array = np.clip((blue_emphasized_array + additional_img_gray)/255, 0, 1).astype('float32')
        add = np.stack((added_images_array,) * 3, axis=-1)
        add = add.transpose((2, 0, 1))
        sample['field'] = add
        if self.transform:
            samples = self.transform(sample)  # Now receives a list of samples
        else:
            samples = [sample]  # If no transform, wrap the original sample into a list
        i = 0
        for s in samples:
            s['case_name'] = self.sample_list[idx].strip('\n') + str(i)
            # Determine the domain based on the length of the case_name
            if len(s['case_name']) == 10:
                s['domain'] = 'Waterloo'
            elif len(s['case_name']) == 8:
                s['domain'] = 'ISIC'
            elif len(s['case_name']) == 4:
                s['domain'] = 'PH2'
            else:
                s['domain'] = 'Unknown'  # Optional: Handle unexpected lengths
            i += 1

        if self.split == "train":
            return samples
        else:
            return samples[0]