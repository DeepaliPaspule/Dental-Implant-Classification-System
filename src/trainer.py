import logging
import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config=None):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer for training
            device: Device to run the training on (cuda/cpu)
            config: Configuration object with training parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Create checkpoint directory
        os.makedirs('models/checkpoints', exist_ok=True)

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)  # Shape: (batch_size, 2, 5)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)  # Shape: (batch_size, 2, 5)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # Log batch results
                if batch_idx % 10 == 0:
                    logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                    
            except Exception as e:
                logging.error(f"Error in training: {str(e)}")
                continue
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(self.train_loader)
        logging.info(f'Epoch {epoch} - Average Training Loss: {avg_loss:.6f}')
        return avg_loss

    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                try:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    
                except Exception as e:
                    logging.error(f"Error in validation: {str(e)}")
                    continue
        
        # Calculate average validation loss
        avg_loss = total_loss / len(self.val_loader)
        logging.info(f'Validation Loss: {avg_loss:.6f}')
        return avg_loss

    def train(self, num_epochs):
        """Train the model for multiple epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logging.info(f'\nEpoch {epoch}/{num_epochs}')
            try:
                # Train for one epoch
                train_loss = self.train_epoch(epoch)
                
                # Validate
                val_loss = self.validate()
                
                # Save checkpoint if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                else:
                    self.save_checkpoint(epoch, val_loss, is_best=False)
                
            except Exception as e:
                logging.error(f"Error in epoch {epoch}: {str(e)}")
                continue

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = f'models/checkpoints/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best validation loss
        if is_best:
            best_model_path = 'models/checkpoints/best_model.pth'
            torch.save(checkpoint, best_model_path)
            logging.info(f'Best model saved: {best_model_path}')
        
        logging.info(f'Checkpoint saved: {checkpoint_path}')