import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from config import Config

class DentalImplantDataset(Dataset):
    def __init__(self, root_dir, category, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.images_dir = os.path.join(root_dir, category, 'images')
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Get all image files recursively
        self.image_files = []
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    self.image_files.append(full_path)
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")
            
        logging.info(f"Found {len(self.image_files)} images for category {category}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise
        
        if self.transform:
            image = self.transform(image)
        
        category_idx = Config.CATEGORIES.index(self.category)
        
        return {
            'image': image,
            'label': category_idx,
            'path': img_path
        }