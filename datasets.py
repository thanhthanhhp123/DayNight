import torch
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DayNightDataset(Dataset):
    def __init__(self, day_dir, night_dir, transform=None):
        self.day_paths = sorted([os.path.join(day_dir, img) for img in os.listdir(day_dir)])
        self.night_paths = sorted([os.path.join(night_dir, img) for img in os.listdir(night_dir)])

        self.transform = transforms.Compose([
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
    

def denormalize(tensor):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    return tensor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    day_dir = 'day_night_images/train/day'
    night_dir = 'day_night_images/train/night'
    
    dataset = DayNightDataset(day_dir, night_dir)

    day_img, night_img = dataset[0]
    day_img = denormalize(day_img)
    night_img = denormalize(night_img)

    plt.subplot(1, 2, 1)
    plt.imshow(day_img.permute(1, 2, 0))
    plt.title('Day')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(night_img.permute(1, 2, 0))
    plt.title('Night')
    plt.axis('off')

    plt.show()