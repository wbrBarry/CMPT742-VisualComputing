import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
import random
import torchvision.transforms as transforms

from PIL import Image, ImageOps

#import any other libraries you need below this line
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom, rotate

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]."""
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, augment_data=True):
    ##########################inputs##################################
    #data_dir(string) - directory of the data#########################
    #size(int) - size of the images you want to use###################
    #train(boolean) - train data or test data#########################
    #train_test_split(float) - the portion of the data for training###
    #augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.augment_data = augment_data

        # Load all file names
        self.images = sorted([os.path.join(data_dir, 'scans', file) for file in os.listdir(os.path.join(data_dir, 'scans'))])
        self.masks = sorted([os.path.join(data_dir, 'labels', file) for file in os.listdir(os.path.join(data_dir, 'labels'))])
        
        # Split dataset
        self.dataset_size = len(self.images)
        self.train_size = int(train_test_split * self.dataset_size)
        if self.train:
            self.images = self.images[:self.train_size]
            self.masks = self.masks[:self.train_size]
        else:
            self.images = self.images[self.train_size:]
            self.masks = self.masks[self.train_size:]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images to [0, 1]
        ])

    def __getitem__(self, idx):
        # Load image and mask from index idx of your data
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Convert PIL image to numpy array
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 5)
            if augment_mode == 0:
                # Flip image vertically
                image_np = np.flipud(image_np)
                mask_np = np.flipud(mask_np)
            elif augment_mode == 1:
                # Flip image horizontally
                image_np = np.fliplr(image_np)
                mask_np = np.fliplr(mask_np)
            elif augment_mode == 2:
                # Zoom image
                # Randomly zooming into the images and masks
                zoom_factor = np.random.uniform(1, 1.5)
                image_np = zoom(image_np, zoom_factor)
                mask_np = zoom(mask_np, zoom_factor, order=0)
            elif augment_mode == 3:
                # Rotate image
                angle = np.random.randint(-180, 180)
                image_np = rotate(image_np, angle, reshape=False, order=1)
                mask_np = rotate(mask_np, angle, reshape=False, order=0)
            elif augment_mode == 4:
                # Elastic deformation
                image_np = elastic_transform(image_np, alpha=image_np.shape[1] * 2, sigma=image_np.shape[1] * 0.08)
                mask_np = elastic_transform(mask_np, alpha=mask_np.shape[1] * 2, sigma=mask_np.shape[1] * 0.08)

        # Convert numpy arrays back to PIL images
        image = Image.fromarray(image_np)
        mask = Image.fromarray(mask_np)

        # Apply transformations and convert to tensor
        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images)