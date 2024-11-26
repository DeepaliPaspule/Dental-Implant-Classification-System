import logging
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision import transforms
import os
from datetime import datetime

from config import Config
from dataset import DentalImplantDataset
from model import DentalImplantModel
from trainer import Trainer

def setup_logging():
    """Set up logging configuration."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_transforms():
    """Get image transformations."""
    return transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_data_loaders():
    """Create data loaders for training and validation."""
    print("Getting data loaders...")
    transform = get_transforms()
    
    # Create datasets for each category
    datasets = []
    for category in Config.CATEGORIES:
        dataset = DentalImplantDataset(
            root_dir=Config.DATA_DIR,
            category=category,
            transform=transform
        )
        datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(
        combined_dataset, 
        [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    print("Starting main function")
    
    # Setup logging
    setup_logging()
    print("Logging setup complete")
    
    logging.info("Starting training pipeline")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders()
    print("Data loaders created")
    
    # Initialize model
    model = DentalImplantModel(num_classes=len(Config.CATEGORIES))
    model = model.to(device)
    print("Model initialized")
    
    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=Config
    )
    print("Trainer initialized")
    
    # Start training
    trainer.train(num_epochs=Config.NUM_EPOCHS)
    logging.info("Training completed")

if __name__ == "__main__":
    print("Starting script...")
    print("Config imported")
    print("Dataset imported")
    print("Model imported")
    print("Trainer imported")
    print("Script started")
    main()