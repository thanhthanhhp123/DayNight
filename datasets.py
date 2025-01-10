import torch
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DayNightDataset(Dataset):
    def __init__(self, day_dir, night_dir, transform=None):
        self.day_paths = sorted([os.path.join(day_dir, img) for img in os.listdir(day_dir)])
        self.night_paths = sorted([os.path.join(night_dir, img) for img in os.listdir(night_dir)])

        self.transform = transform.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ]) if transform is None else transform

    def __len__(self):
        return max(len(self.day_paths), len(self.night_paths))
    
    def __getitem__(self, idx):
        # Lấy ảnh ngày theo index
        day_img = Image.open(self.day_paths[idx % len(self.day_paths)]).convert('RGB')
        # Lấy ảnh đêm ngẫu nhiên
        night_path = random.choice(self.night_paths)
        night_img = Image.open(night_path).convert('RGB')
        
        if self.transform:
            day_img = self.transform(day_img)
            night_img = self.transform(night_img)
            
        return day_img, night_img