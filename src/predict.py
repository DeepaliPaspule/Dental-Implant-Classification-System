import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from model import DentalImplantModel
from config import Config

class DentalImplantPredictor:
    def __init__(self, model_path='models/checkpoints/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = DentalImplantModel(num_classes=len(Config.CATEGORIES))
        self.model.load_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                class_id = int(torch.argmax(probabilities))
                confidence = float(probabilities[class_id])
            
            return {
                'class': Config.CATEGORIES[class_id],
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error predicting image {image_path}: {str(e)}")
            raise