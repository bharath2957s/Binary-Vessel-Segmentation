import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DRIVEDataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, '1st_manual')
        self.roi_dir = os.path.join(root_dir, split, 'mask')
        
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        if self.split == 'training':
            mask_name = img_name.replace('_training.tif', '_manual1.gif')
        else:
            mask_name = img_name.replace('_test.tif', '_manual1.gif')
        
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        
        # Load ROI mask
        if self.split == 'training':
            roi_name = img_name.replace('_training.tif', '_training_mask.gif')
        else:
            roi_name = img_name.replace('_test.tif', '_test_mask.gif')
            
        roi_path = os.path.join(self.roi_dir, roi_name)
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        roi = (roi > 0).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0).float(),
            'roi': torch.tensor(roi, dtype=torch.float32).unsqueeze(0),
            'filename': img_name
        }

def get_transforms(split='training'):
    if split == 'training':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])