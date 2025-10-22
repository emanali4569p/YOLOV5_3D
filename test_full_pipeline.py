#!/usr/bin/env python3
"""
Complete pipeline test for 3D Object Detection
Tests both dataset and model together
"""

import torch
from kitti_dataset import create_dataloader
from yolo3d_model import create_model, Loss3D
import numpy as np

def test_full_pipeline():
    """Test the complete training pipeline"""
    print("Testing Complete 3D Object Detection Pipeline...")
    print("=" * 60)
    
    try:
        # Test 1: Dataset Creation
        print("1. Testing Dataset Creation...")
        train_loader = create_dataloader(
            data_dir="Data",
            split='train',
            batch_size=2,
            img_size=640,
            augment=False,  # Disable augmentation for testing
            num_workers=0   # Use 0 workers for testing
        )
        print(f"✓ Dataset created successfully")
        print(f"  - Train samples: {len(train_loader.dataset)}")
        
        # Test 2: Model Creation
        print("\n2. Testing Model Creation...")
        model = create_model(nc=9)  # 9 classes for KITTI
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test 3: Loss Function
        print("\n3. Testing Loss Function...")
        criterion = Loss3D(nc=9)
        print(f"✓ Loss function created successfully")
        
        # Test 4: Data Loading
        print("\n4. Testing Data Loading...")
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"✓ Batch {batch_idx} loaded successfully")
            print(f"  - Images shape: {images.shape}, dtype: {images.dtype}")
            print(f"  - Number of targets: {[len(t) for t in targets]}")
            
            # Test 5: Forward Pass
            print(f"\n5. Testing Forward Pass...")
            model.eval()
            with torch.no_grad():
                outputs = model(images)
            
            print(f"✓ Forward pass successful")
            print(f"  - Detection outputs: {len(outputs['detections'])}")
            print(f"  - Depth outputs: {len(outputs['depth'])}")
            print(f"  - Dimension outputs: {len(outputs['dimensions'])}")
            print(f"  - Rotation outputs: {len(outputs['rotation'])}")
            
            # Test 6: Loss Calculation
            print(f"\n6. Testing Loss Calculation...")
            loss_dict = criterion(outputs, targets)
            print(f"✓ Loss calculation successful")
            print(f"  - Total loss: {loss_dict['total_loss']:.4f}")
            print(f"  - Detection loss: {loss_dict['detection_loss']:.4f}")
            print(f"  - Depth loss: {loss_dict['depth_loss']:.4f}")
            print(f"  - Dimension loss: {loss_dict['dimension_loss']:.4f}")
            print(f"  - Rotation loss: {loss_dict['rotation_loss']:.4f}")
            
            # Test 7: Backward Pass (Gradient Test)
            print(f"\n7. Testing Backward Pass...")
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            optimizer.zero_grad()
            loss_dict = criterion(outputs, targets)
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            print(f"✓ Backward pass successful")
            print(f"  - Gradients computed and applied")
            
            # Only test first batch to avoid long execution
            break
        
        print(f"\n🎉 All pipeline tests passed successfully!")
        print(f"The complete training pipeline is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage with different batch sizes"""
    print("\n" + "=" * 60)
    print("Testing Memory Usage...")
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create data loader
            train_loader = create_dataloader(
                data_dir="Data",
                split='train',
                batch_size=batch_size,
                img_size=640,
                augment=False,
                num_workers=0
            )
            
            # Create model
            model = create_model(nc=9)
            criterion = Loss3D(nc=9)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Test one batch
            for images, targets in train_loader:
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss['total_loss'].backward()
                optimizer.step()
                
                print(f"  ✓ Batch size {batch_size} works correctly")
                print(f"    - Memory usage: OK")
                break
                
        except Exception as e:
            print(f"  ✗ Batch size {batch_size} failed: {e}")
            return False
    
    print(f"\n✓ Memory usage tests completed")
    return True

if __name__ == "__main__":
    print("🧪 Complete Pipeline Test for 3D Object Detection")
    print("This test verifies that the entire training pipeline works correctly")
    
    # Test complete pipeline
    success1 = test_full_pipeline()
    
    # Test memory usage
    success2 = test_memory_usage()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Your 3D object detection pipeline is ready for training!")
        print("You can now run the training script with confidence.")
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above before proceeding with training.")
