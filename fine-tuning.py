import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class CCTVDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform = None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_image_list = os.listdir(hr_dir)
        self.lr_image_list = os.listdir(lr_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.hr_image_list)
    
    def __getitem__(self, index):
        hr_image_path = os.path.join(self.hr_dir, self.hr_image_list[index])
        lr_image_path = os.path.join(self.lr_dir, self.lr_image_list[index])
        
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')
        
        if self.transform:
            hr_image, lr_image = self.transform(hr_image, lr_image)
            
        return hr_image, lr_image