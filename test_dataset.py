#!/usr/bin/env python3
"""
Test script to verify the KITTI dataset works correctly
"""

import torch
from kitti_dataset import KITTIDataset, create_dataloader
import numpy as np

def test_dataset():
    """Test the KITTI dataset functionality"""
    print("Testing KITTI Dataset...")
    
    try:
        # Test dataset creation
        dataset = KITTIDataset("Data", split='train', img_size=640, augment=False)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test single sample loading
        if len(dataset) > 0:
            image, targets = dataset[0]
            print(f"✓ Single sample loaded:")
            print(f"  - Image shape: {image.shape}, type: {type(image)}")
            print(f"  - Image dtype: {image.dtype}")
            print(f"  - Targets shape: {targets.shape}, type: {type(targets)}")
            print(f"  - Number of objects: {len(targets)}")
            
            # Verify image is a tensor
            if isinstance(image, torch.Tensor):
                print("✓ Image is properly converted to tensor")
            else:
                print("✗ ERROR: Image is not a tensor!")
                return False
                
            # Test data loader
            print("\nTesting DataLoader...")
            dataloader = create_dataloader("Data", split='train', batch_size=2, num_workers=0)
            
            for batch_idx, (images, targets) in enumerate(dataloader):
                print(f"✓ Batch {batch_idx} loaded:")
                print(f"  - Images shape: {images.shape}, dtype: {images.dtype}")
                print(f"  - Number of targets: {[len(t) for t in targets]}")
                
                # Verify batch images are tensors
                if isinstance(images, torch.Tensor):
                    print("✓ Batch images are properly converted to tensors")
                else:
                    print("✗ ERROR: Batch images are not tensors!")
                    return False
                
                if batch_idx >= 2:  # Test only first 3 batches
                    break
                    
        print("\n✓ All tests passed! Dataset is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n🎉 Dataset test completed successfully!")
    else:
        print("\n❌ Dataset test failed!")
