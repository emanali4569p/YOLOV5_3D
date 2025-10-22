import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

from kitti_dataset import create_dataloader
from yolo3d_model import create_model, Loss3D

class Trainer3D:
    """3D Object Detection Trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.setup_directories()
        
        # Setup data
        self.setup_data()
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Setup loss function
        self.criterion = Loss3D(nc=len(self.train_dataset.dataset.classes))
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_directories(self):
        """Setup training directories"""
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.results_dir = Path("results")
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def setup_data(self):
        """Setup data loaders"""
        self.train_dataset = create_dataloader(
            data_dir=self.config['data_dir'],
            split='train',
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            augment=True
        )
        
        self.val_dataset = create_dataloader(
            data_dir=self.config['data_dir'],
            split='test',
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            augment=False
        )
        
        print(f"Train samples: {len(self.train_dataset.dataset)}")
        print(f"Validation samples: {len(self.val_dataset.dataset)}")
    
    def setup_model(self):
        """Setup model"""
        self.model = create_model(nc=len(self.train_dataset.dataset.classes))
        self.model.to(self.device)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Use SGD for better learning with momentum
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            momentum=0.9,
            weight_decay=self.config.get('weight_decay', 1e-4),
            nesterov=True  # Nesterov momentum for better convergence
        )
        
        # Use ReduceLROnPlateau for adaptive learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_dataset, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), 
                                     epoch * len(self.train_dataset) + batch_idx)
        
        return total_loss / num_batches
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_dataset:
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        best_loss = float('inf')
        start_epoch = 0
        
        # Load checkpoint if resuming
        if self.config.get('resume'):
            start_epoch, best_loss = self.load_checkpoint(self.config['resume'])
            print(f"Resumed training from epoch {start_epoch+1}")
        
        for epoch in range(start_epoch, self.config['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            # Log metrics
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping with more patience
            if epoch > 20 and val_loss > best_loss * 1.2:
                print("Early stopping triggered")
                break
        
        print("Training completed!")
        self.writer.close()

def create_config():
    """Create training configuration"""
    return {
        'data_dir': 'Data',
        'img_size': 416,  # Reduced to save memory
        'batch_size': 2,  # Reduced to save memory
        'epochs': 100,  # Increased for better learning
        'learning_rate': 0.005,  # Optimized learning rate
        'weight_decay': 1e-4,
        'resume': None
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YOLOv5 3D Object Detection')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_config()
    
    # Override resume path if provided
    if args.resume:
        config['resume'] = args.resume
    
    # Create trainer
    trainer = Trainer3D(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()