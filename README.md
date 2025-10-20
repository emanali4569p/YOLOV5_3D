# Enhanced YOLOv5 Model for 3D Object Detection on KITTI Dataset

## Overview

This project presents an enhanced YOLOv5 model specifically designed for 3D object detection using the KITTI dataset. The model is capable of:

- Detecting objects in 2D images
- Estimating 3D positions of objects
- Estimating object dimensions (height, width, length)
- Estimating object rotation angles
- Estimating depth and distance

## Key Features

### 🚗 3D Object Detection
- Support for all KITTI classes (cars, pedestrians, cyclists, etc.)
- Accurate estimation of 3D positions and dimensions
- Handling rotation angles and depth

### 🎯 Enhanced Model
- YOLOv5 architecture optimized for 3D detection
- Specialized prediction heads for depth, dimensions, and rotation
- Improvements in detection accuracy and speed

### 📊 Advanced Data Processing
- Automatic data splitting (80% training, 20% testing)
- Advanced data transformations (Augmentation)
- Processing of calibration and LiDAR files

## Project Structure

```
3D OBJ DETEC/
├── Data/                    # KITTI dataset
│   ├── image_2/            # RGB images
│   ├── label_2/            # 3D labels
│   ├── calib/              # Calibration files
│   └── velodyne/           # LiDAR data
├── kitti_dataset.py        # Dataset class
├── yolo3d_model.py         # 3D YOLOv5 model
├── train_3d.py            # Training script
├── evaluate_3d.py         # Evaluation script
├── evaluate_difficulty.py # Difficulty-based evaluation
├── analyze_data.py         # Data analysis
├── config.yaml            # Configuration file
├── requirements.txt       # Requirements
└── README.md             # This file
```

## Installation and Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Setup Data

Ensure the KITTI dataset is located in the `Data/` folder with the following structure:

```
Data/
├── image_2/     # Images (000000.png, 000001.png, ...)
├── label_2/     # Labels (000000.txt, 000001.txt, ...)
├── calib/       # Calibration (000000.txt, 000001.txt, ...)
└── velodyne/    # LiDAR (000000.bin, 000001.bin, ...)
```

### 3. Modify Settings

You can modify settings in the `config.yaml` file:

```yaml
# Data
data_dir: "Data"
img_size: 640
batch_size: 8

# Training
epochs: 100
learning_rate: 0.001
```

## Usage

### Training

```bash
# Basic training
python train_3d.py

# Training with custom settings
python train_3d.py --config config.yaml

# Resume training from checkpoint
python train_3d.py --resume checkpoints/best_model.pth
```

### Evaluation

```bash
# Evaluate trained model
python evaluate_3d.py --model checkpoints/best_model.pth

# Evaluate with visualization
python evaluate_3d.py --model checkpoints/best_model.pth --visualize

# Save evaluation results
python evaluate_3d.py --model checkpoints/best_model.pth --save_results results.txt
```

### Difficulty-based Evaluation

```bash
# Advanced evaluation by difficulty levels
python evaluate_difficulty.py --model checkpoints/best_model.pth

# Or use the easy runner
python run_difficulty_eval.py
```

### Test Dataset

```bash
python kitti_dataset.py
```

## Model and Architecture

### Basic Architecture

1. **Backbone**: Feature extraction network based on YOLOv5
2. **Neck**: Feature Pyramid Network (FPN) for multi-level feature fusion
3. **Head**: Specialized prediction heads for 3D detection

### Outputs

The model produces:

- **2D Detection**: Bounding boxes, confidence, and class
- **Depth**: Distance estimation for objects
- **3D Dimensions**: Height, width, length
- **Rotation**: Rotation angle around Y-axis

### Loss Function

```python
Total Loss = Detection Loss + 0.1 × Depth Loss + 0.1 × Dimension Loss + 0.1 × Rotation Loss
```

## Expected Results

### Performance Metrics

- **Precision**: > 0.85
- **Recall**: > 0.80
- **F1 Score**: > 0.82
- **Average Inference Time**: < 50ms

### Supported Object Classes

1. **Car** - Cars
2. **Van** - Small trucks
3. **Truck** - Large trucks
4. **Pedestrian** - Pedestrians
5. **Person_sitting** - Sitting persons
6. **Cyclist** - Cyclists
7. **Tram** - Trams
8. **Misc** - Other objects
9. **DontCare** - Unimportant objects

## Improvements and Development

### Applied Improvements

1. **Data Processing**:
   - Advanced data transformations
   - Calibration file processing
   - 3D to 2D projection

2. **Model**:
   - Specialized prediction heads
   - Improved loss function
   - 3D information processing

3. **Training**:
   - Learning rate scheduling
   - Gradient clipping
   - Early stopping

### Development Suggestions

1. **Improve Depth Accuracy**:
   - Use LiDAR data directly
   - Add separate depth estimation network

2. **Improve Performance**:
   - Optimize Non-Maximum Suppression
   - Add Multi-Scale Training

3. **Support Additional Classes**:
   - Add new classes from other datasets
   - Train on multiple datasets

## Troubleshooting

### Common Issues

1. **Data Loading Error**:
   - Check file paths
   - Verify all required files exist

2. **Memory Issues**:
   - Reduce batch size
   - Use mixed precision training

3. **Slow Training**:
   - Ensure GPU usage
   - Reduce number of workers

### Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Batch Size**: Adjust according to available GPU memory
3. **Data Optimization**: Use SSD for data storage
4. **Optimize Data**: Reduce num_workers if issues occur

## Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## References

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [3D Object Detection Papers](https://paperswithcode.com/task/3d-object-detection)

## Support

If you encounter any issues or have questions, please:

1. Open an Issue on GitHub
2. Review documentation
3. Check known issues

---

**Note**: This project is in active development. Results may vary depending on data quality and training settings.