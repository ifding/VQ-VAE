import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import albumentations as albu
from albumentations.pytorch import ToTensor 


def get_valid_transforms(crop_size=320):
    return albu.Compose([
        albu.Resize(crop_size, crop_size),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensor(),
    ])

def light_training_transforms(crop_size=320):
    return albu.Compose([
        albu.RandomResizedCrop(height=crop_size, width=crop_size),
        albu.OneOf(
            [
                albu.Transpose(),
                albu.VerticalFlip(),
                albu.HorizontalFlip(),
                albu.RandomRotate90(),
                albu.NoOp()
            ], p=1.0),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensor(),
    ])

def medium_training_transforms(crop_size=320):
    return albu.Compose([
        albu.RandomResizedCrop(height=crop_size, width=crop_size),
        albu.OneOf(
            [
                albu.Transpose(),
                albu.VerticalFlip(),
                albu.HorizontalFlip(),
                albu.RandomRotate90(),
                albu.NoOp()
            ], p=1.0),
        albu.OneOf(
            [
                albu.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                albu.NoOp()
            ], p=1.0),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensor(),
    ])


def heavy_training_transforms(crop_size=320):
    return albu.Compose([
        albu.RandomResizedCrop(height=crop_size, width=crop_size),
        albu.OneOf(
            [
                albu.Transpose(),
                albu.VerticalFlip(),
                albu.HorizontalFlip(),
                albu.RandomRotate90(),
                albu.NoOp()
            ], p=1.0),
        albu.ShiftScaleRotate(p=0.75),
        albu.OneOf(
            [
                albu.CoarseDropout(max_holes=16, max_height=16, max_width=16),
                albu.NoOp()
            ], p=1.0),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensor(),
    ])

def get_training_trasnforms(transforms_type):
    if transforms_type == 'light':
        return(light_training_transforms())
    elif transforms_type == 'medium':
        return(medium_training_transforms())
    elif transforms_type == 'heavy':
        return(heavy_training_transforms())
    else:
        raise NotImplementedError("Not implemented transformation configuration")


class SegCTDataset(Dataset):
    """Covid XRay dataset."""
    def __init__(
            self, 
            txt, 
            images_dir = '../datasets/2d_images/', 
            masks_dir = '../datasets/2d_masks/', 
            augmentation=None, 
    ):
        self.IMAGE_LIB = images_dir
        self.MASK_LIB = masks_dir

        self.images = np.loadtxt(txt, dtype=str)

        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.images[idx]
        img = cv2.imread(self.IMAGE_LIB + image_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
       
        img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
        #img = Image.fromarray(np.uint8(img * 255)).convert('L')
        img = np.asarray(img)
        mask = Image.open(self.MASK_LIB + image_name)
        mask = np.asarray(mask)
        #mask = np.expand_dims(mask, -1)
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
            
        return img, mask

class TestDataset(Dataset):
    """Test dataset."""
    def __init__(
            self, 
            txt, 
            images_dir, 
            size, mean, std, 
    ):
        self.IMAGE_LIB = images_dir
        self.images = np.loadtxt(txt, dtype=str)

        self.transform = albu.Compose(
            [
                albu.Normalize(mean=mean, std=std, p=1),
                albu.Resize(size, size),
                ToTensor(),
            ]
        )        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.images[idx]
        img = cv2.imread(self.IMAGE_LIB + image_name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  
        img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
        #img = Image.fromarray(np.uint8(img * 255)).convert('L')
        img = np.asarray(img)
        img = self.transform(image=img)["image"]  
        return img
