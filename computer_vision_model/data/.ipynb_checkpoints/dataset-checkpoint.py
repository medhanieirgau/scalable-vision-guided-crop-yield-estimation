import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

class CropYieldDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        images = df.iloc[:,0]
        self.labels = df.iloc[:,1]
        self.image_paths = [os.path.join("/scr/mirgau/crop_yield_data/", img) for img in images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        log_label = np.log(label+1)
        #label = 1 if self.labels[idx] >= 1 else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)
    
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        
    transforms.RandomHorizontalFlip(),    
    transforms.RandomRotation(10),        
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),        
    transforms.ToTensor(),                
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class Maize(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        df = pd.read_csv(csv_path)
        cce_ids = df['cce_id']
        self.labels = df['Avg_yield_mt_ha']
        self.image_paths = [f"/scr/mirgau/q_field_photo/q_field_photo_{cce_id}.jpg" for cce_id in cce_ids]

        #images = df.iloc[:,0]
        #self.labels = df.iloc[:,1]
        #self.image_paths = [os.path.join("/scr/mirgau/q_field_photo/", img) for img in images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        #log_label = np.log(label+1)
        #label = 1 if self.labels[idx] >= 1 else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)



