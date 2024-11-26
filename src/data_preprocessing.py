import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DentalImplantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Categories mapping
        self.categories = {
            'endosteal': 0,
            'subperiosteal': 1,
            'transosteal': 2,
            'zygomatic': 3
        }
        
        # Load all data
        self._load_data()
    
    def _load_data(self):
        for category in self.categories.keys():
            category_path = os.path.join(self.root_dir, category)
            images_path = os.path.join(category_path, 'images')
            labels_path = os.path.join(category_path, 'labels')
            
            # Get all image files
            image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(images_path, img_file)
                label_file = os.path.join(labels_path, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                
                if os.path.exists(label_file):
                    self.data.append({
                        'image_path': img_path,
                        'label_path': label_file,
                        'category': self.categories[category]
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load label data
        with open(item['label_path'], 'r') as f:
            label_data = f.read().strip()
        
        # Convert label data to tensor
        # Assuming label data is space-separated values
        label_values = torch.tensor([float(x) for x in label_data.split()])
        
        return {
            'image': image,
            'label_data': label_values,
            'category': item['category']
        }