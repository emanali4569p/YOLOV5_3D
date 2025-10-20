import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import yaml
import json
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class KITTIDataset(Dataset):
    """KITTI dataset class for 3D object detection"""
    
    def __init__(self, data_dir: str, split: str = 'train', img_size: int = 640, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # KITTI classes
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
                       'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data indices
        self.indices = self.load_indices()
        
        # Data transformations
        self.transform = self.get_transforms()
        
        print(f"Loaded {len(self.indices)} samples for {split} split")
    
    def load_indices(self):
        """Load data indices based on split with label-aware filtering and fixed seed."""
        # Collect indices from image filenames and ensure matching label exists
        image_dir = self.data_dir / "image_2"
        label_dir = self.data_dir / "label_2"
        image_files = sorted(image_dir.glob("*.png"))
        valid_indices = []
        for img_path in image_files:
            try:
                stem = img_path.stem  # e.g., '000123'
                idx = int(stem)
            except Exception:
                continue
            if (label_dir / f"{idx:06d}.txt").exists():
                valid_indices.append(idx)

        # Deterministic shuffle with fixed seed
        rng = np.random.RandomState(42)
        rng.shuffle(valid_indices)

        # Split (80/20)
        split_idx = int(len(valid_indices) * 0.8)
        if self.split == 'train':
            return valid_indices[:split_idx]
        else:
            return valid_indices[split_idx:]
    
    def get_transforms(self):
        """Get data transformations"""
        if self.split == 'train' and self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        actual_idx = self.indices[idx]
        
        # Load image
        image_path = self.data_dir / "image_2" / f"{actual_idx:06d}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.data_dir / "label_2" / f"{actual_idx:06d}.txt"
        targets = self.load_labels(label_path)
        
        # Apply transformations
        if self.transform and len(targets) > 0:
            # Prepare bboxes and labels for augmentation
            bboxes = []
            class_labels = []
            
            for target in targets:
                if len(target) >= 5:
                    class_id = int(target[0])
                    cx, cy, w, h = target[1:5].tolist()
                    
                    # Convert to pixel coordinates for augmentation
                    x1 = (cx - w/2) * self.img_size
                    y1 = (cy - h/2) * self.img_size
                    x2 = (cx + w/2) * self.img_size
                    y2 = (cy + h/2) * self.img_size
                    
                    bboxes.append([x1, y1, x2, y2])
                    class_labels.append(class_id)
            
            if bboxes:
                try:
                    transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    image = transformed['image']
                    
                    # Convert back to normalized coordinates
                    new_targets = []
                    for i, bbox in enumerate(transformed['bboxes']):
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2 / self.img_size
                        cy = (y1 + y2) / 2 / self.img_size
                        w = (x2 - x1) / self.img_size
                        h = (y2 - y1) / self.img_size
                        
                        # Keep original target but update bbox coordinates
                        original_target = targets[i].clone()
                        original_target[1:5] = torch.tensor([cx, cy, w, h])
                        new_targets.append(original_target)
                    
                    targets = torch.stack(new_targets) if new_targets else torch.zeros((0, 16))
                except:
                    # If augmentation fails, use original data
                    pass
        
        return image, targets
    
    def load_labels(self, label_path):
        """Load and process labels"""
        targets = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        class_name = parts[0]
                        
                        # Skip DontCare class
                        if class_name == 'DontCare':
                            continue
                        
                        if class_name in self.class_to_idx:
                            class_id = self.class_to_idx[class_name]
                            
                            # Extract 2D bounding box
                            bbox_2d = [float(x) for x in parts[4:8]]
                            
                            # Extract 3D information
                            dimensions = [float(x) for x in parts[8:11]]  # h, w, l
                            location = [float(x) for x in parts[11:14]]   # x, y, z
                            rotation_y = float(parts[14])
                            
                            # Convert 2D bbox to normalized coordinates
                            x1, y1, x2, y2 = bbox_2d
                            cx = (x1 + x2) / 2 / 1242  # Normalize by image width
                            cy = (y1 + y2) / 2 / 375   # Normalize by image height
                            w = (x2 - x1) / 1242
                            h = (y2 - y1) / 375
                            
                            # Create target tensor
                            target = torch.tensor([
                                class_id, cx, cy, w, h,
                                float(parts[1]),  # truncated
                                int(parts[2]),    # occluded
                                float(parts[3]),  # alpha
                                *dimensions,      # 3D dimensions
                                *location,        # 3D location
                                rotation_y        # rotation
                            ])
                            
                            targets.append(target)
        
        if targets:
            return torch.stack(targets)
        else:
            # Return empty tensor if no valid targets
            return torch.zeros((0, 16))
    
    def collate_fn(self, batch):
        """Custom collate function for batching"""
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, targets

def create_dataloader(data_dir: str, split: str = 'train', batch_size: int = 8, 
                     img_size: int = 640, augment: bool = True, num_workers: int = 4):
    """Create data loader"""
    dataset = KITTIDataset(data_dir, split, img_size, augment)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    return dataloader

def visualize_sample(dataset, idx: int = 0):
    """Visualize a sample from the dataset"""
    image, targets = dataset[idx]
    
    # Convert image back to display format
    image_np = image.permute(1, 2, 0).numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    
    # Draw bounding boxes
    for target in targets:
        if len(target) >= 5:
            class_id = int(target[0])
            cx, cy, w, h = target[1:5].tolist()
            
            # Convert to pixel coordinates
            x1 = (cx - w/2) * dataset.img_size
            y1 = (cy - h/2) * dataset.img_size
            x2 = (cx + w/2) * dataset.img_size
            y2 = (cy + h/2) * dataset.img_size
            
            # Draw rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add class label
            ax.text(x1, y1-5, dataset.classes[class_id], 
                   color='red', fontsize=10, weight='bold')
    
    ax.set_title(f'Sample {idx} - {len(targets)} objects')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def test_dataset():
    """Test dataset functionality"""
    print("Testing KITTI dataset...")
    
    # Create dataset
    dataset = KITTIDataset("Data", split='train')
    
    # Test loading
    if len(dataset) > 0:
        image, targets = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Number of objects: {len(targets)}")
        
        if len(targets) > 0:
            print(f"First target: {targets[0]}")
    
    # Test data loader
    dataloader = create_dataloader("Data", split='train', batch_size=2)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of targets: {[len(t) for t in targets]}")
        
        if batch_idx >= 2:  # Test only first 3 batches
            break
    
    print("Dataset test completed!")

if __name__ == "__main__":
    test_dataset()