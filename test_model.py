#!/usr/bin/env python3
"""
Test script to verify the YOLOv5 3D model works correctly
"""

import torch
from yolo3d_model import create_model, Loss3D
import numpy as np

def test_model():
    """Test the YOLOv5 3D model functionality"""
    print("Testing YOLOv5 3D Model...")
    
    try:
        # Create model
        model = create_model(nc=9)  # 9 classes for KITTI
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with different batch sizes
        batch_sizes = [1, 2, 4]
        img_size = 640
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create input tensor
            x = torch.randn(batch_size, 3, img_size, img_size)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(x)
            
            print(f"  ✓ Forward pass successful")
            print(f"  - Input shape: {x.shape}")
            print(f"  - Detection outputs: {len(outputs['detections'])}")
            print(f"  - Depth outputs: {len(outputs['depth'])}")
            print(f"  - Dimension outputs: {len(outputs['dimensions'])}")
            print(f"  - Rotation outputs: {len(outputs['rotation'])}")
            
            # Check output shapes
            for i, det in enumerate(outputs['detections']):
                print(f"    Detection {i}: {det.shape}")
            for i, depth in enumerate(outputs['depth']):
                print(f"    Depth {i}: {depth.shape}")
            for i, dim in enumerate(outputs['dimensions']):
                print(f"    Dimensions {i}: {dim.shape}")
            for i, rot in enumerate(outputs['rotation']):
                print(f"    Rotation {i}: {rot.shape}")
        
        # Test loss function
        print(f"\nTesting Loss Function...")
        loss_fn = Loss3D(nc=9)
        
        # Create dummy targets (empty for testing)
        dummy_targets = [torch.zeros(0, 16) for _ in range(batch_sizes[-1])]
        
        # Test with last batch size
        x = torch.randn(batch_sizes[-1], 3, img_size, img_size)
        with torch.no_grad():
            outputs = model(x)
            loss = loss_fn(outputs, dummy_targets)
        
        print(f"✓ Loss calculation successful")
        print(f"  - Total loss: {loss['total_loss']:.4f}")
        print(f"  - Detection loss: {loss['detection_loss']:.4f}")
        print(f"  - Depth loss: {loss['depth_loss']:.4f}")
        print(f"  - Dimension loss: {loss['dimension_loss']:.4f}")
        print(f"  - Rotation loss: {loss['rotation_loss']:.4f}")
        
        print(f"\n✓ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_cuda():
    """Test model with CUDA if available"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA test")
        return True
    
    print("\nTesting Model with CUDA...")
    
    try:
        # Create model and move to CUDA
        model = create_model(nc=9).cuda()
        
        # Test forward pass on GPU
        x = torch.randn(2, 3, 640, 640).cuda()
        
        with torch.no_grad():
            outputs = model(x)
        
        print(f"✓ CUDA forward pass successful")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Output device: {outputs['detections'][0].device}")
        
        return True
        
    except Exception as e:
        print(f"✗ CUDA ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing YOLOv5 3D Model Architecture")
    print("=" * 50)
    
    # Test basic functionality
    success1 = test_model()
    
    # Test CUDA functionality
    success2 = test_model_with_cuda()
    
    if success1 and success2:
        print("\n🎉 All model tests completed successfully!")
        print("The model is ready for training!")
    else:
        print("\n❌ Some model tests failed!")
        print("Please check the errors above.")
