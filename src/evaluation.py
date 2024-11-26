import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from model import DentalImplantModel
from dataset import DentalImplantDataset
from config import Config

def create_datasets():
    """Create and combine datasets for all categories"""
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_datasets = []
    valid_categories = []

    # Verify data directory structure first
    if not Config.check_data_directory():
        raise ValueError("Invalid data directory structure")

    for category in Config.CATEGORIES:
        try:
            dataset = DentalImplantDataset(
                root_dir=Config.DATA_DIR,
                category=category,
                transform=transform
            )
            all_datasets.append(dataset)
            valid_categories.append(category)
            print(f"Successfully added dataset for category {category}")
        except Exception as e:
            print(f"Error creating dataset for category {category}: {str(e)}")
    
    if not all_datasets:
        raise ValueError(
            "No valid datasets found. Please check:\n"
            f"1. Data directory exists: {Config.DATA_DIR}\n"
            "2. Category folders exist and contain images\n"
            f"3. Expected categories: {Config.CATEGORIES}\n"
            f"4. Valid categories found: {valid_categories}"
        )
    
    return ConcatDataset(all_datasets)

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CATEGORIES,
                yticklabels=Config.CATEGORIES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Starting evaluation process...")
        Config.print_paths()
        
        # Create dataset
        test_dataset = create_datasets()
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model
        model = DentalImplantModel(num_classes=len(Config.CATEGORIES))
        model.load_pretrained('models/checkpoints/best_model.pth')
        model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()