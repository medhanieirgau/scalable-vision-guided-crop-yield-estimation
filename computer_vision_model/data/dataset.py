import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class CropDataset(Dataset):
    def __init__(self, csv_path, folds, zone2idx, config, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['fold'].isin(folds)]
        cce_ids = df['cce_id'].tolist()
        self.image_paths = [os.path.join(config["paths"]["data_dir"],f"q_field_photo/q_field_photo_{cce_id}.jpg") for cce_id in cce_ids]
        self.labels = df['Avg_yield_mt_ha'].tolist()
        
        self.zone_ids = [zone2idx[z] for z in df["zone"]]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        zone_id = torch.tensor(self.zone_ids[idx], dtype=torch.long)
        return image, label, zone_id
    
    @staticmethod
    def build_transform(is_train, input_size):
        interpol_mode = transforms.InterpolationMode.BICUBIC
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode, antialias=True))
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(transforms.Resize(size, interpolation=interpol_mode, antialias=True),)
        t.append(transforms.CenterCrop(input_size))
        return transforms.Compose(t)

