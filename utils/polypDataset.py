import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from skimage.io import imread

class PolypDataset(Dataset):
    def __init__(self, img_height, img_width, dataset='kvasir', base_dir=None, split='train', transform=None, seed=42 , method=None):
        self.img_height = img_height
        self.img_width = img_width
        self.dataset = dataset
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        self.method = method

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.IMAGES_PATH = os.path.join(self.base_dir, self.dataset, self.split, 'images/')
        self.MASKS_PATH = os.path.join(self.base_dir, self.dataset, self.split, 'masks/')
        print(self.IMAGES_PATH)
        self.train_ids = glob.glob(os.path.join(self.IMAGES_PATH, "*.jpg"))

        print("total {} samples for {}".format(len(self.train_ids), self.split))

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        image_path = self.train_ids[index]
        mask_path = image_path.replace("images", "masks")

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Resize images and masks
        # image = image.resize((self.img_width, self.img_height), Image.BILINEAR)
        # mask = mask.resize((self.img_width, self.img_height), Image.NEAREST)
        if self.method != 'SLAUG':
            image = np.array(image) / 255.0  # Normalize to [0, 1]
        else:
            image = np.array(image) / 255.0

        mask = np.array(mask) > 127  # Binarize mask
        mask = mask.astype(np.float32)  # Convert to float32

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to CxHxW format
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        sample = {'image': image, 'label': mask, 'img_name': index}

        if self.transform:
            sample = self.transform(sample)
            
        if self.method == 'CCSDG' or 'SLAUG':
            return sample['image'],sample['label'],sample['img_name']

        return sample
