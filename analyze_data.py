import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import Counter
import cv2

def analyze_kitti_dataset(data_dir="Data"):
    """Comprehensive analysis of KITTI dataset"""
    
    print("🔍 Analyzing KITTI dataset...")
    
    data_path = Path(data_dir)
    
    # File statistics
    print("\n📁 File statistics:")
    for folder in ["image_2", "label_2", "calib", "velodyne"]:
        folder_path = data_path / folder
        if folder_path.exists():
            files = list(folder_path.glob("*"))
            print(f"  {folder}: {len(files)} files")
        else:
            print(f"  {folder}: not found")
    
    # Analyze labels
    analyze_labels(data_path / "label_2")
    
    # Analyze images
    analyze_images(data_path / "image_2")
    
    # Analyze calibration
    analyze_calibration(data_path / "calib")

def analyze_labels(label_dir):
    """Analyze label files"""
    print("\n🏷️  Label analysis:")
    
    label_files = list(label_dir.glob("*.txt"))
    if not label_files:
        print("  No label files found")
        return
    
    # Class statistics
    class_counts = Counter()
    total_objects = 0
    truncated_objects = 0
    occluded_objects = 0
    
    # Dimension statistics
    dimensions = {'height': [], 'width': [], 'length': []}
    locations = {'x': [], 'y': [], 'z': []}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15:
                    class_name = parts[0]
                    truncated = float(parts[1])
                    occluded = int(parts[2])
                    
                    # 3D dimensions
                    h, w, l = map(float, parts[8:11])
                    dimensions['height'].append(h)
                    dimensions['width'].append(w)
                    dimensions['length'].append(l)
                    
                    # 3D positions
                    x, y, z = map(float, parts[11:14])
                    locations['x'].append(x)
                    locations['y'].append(y)
                    locations['z'].append(z)
                    
                    class_counts[class_name] += 1
                    total_objects += 1
                    
                    if truncated > 0:
                        truncated_objects += 1
                    if occluded > 0:
                        occluded_objects += 1
    
    # Print statistics
    print(f"  Total objects: {total_objects}")
    print(f"  Truncated objects: {truncated_objects} ({truncated_objects/total_objects*100:.1f}%)")
    print(f"  Occluded objects: {occluded_objects} ({occluded_objects/total_objects*100:.1f}%)")
    
    print("\n  Class distribution:")
    for class_name, count in class_counts.most_common():
        percentage = count / total_objects * 100
        print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    # Plot graphs
    plot_class_distribution(class_counts)
    plot_dimensions_distribution(dimensions)
    plot_locations_distribution(locations)

def analyze_images(image_dir):
    """Analyze images"""
    print("\n🖼️  Image analysis:")
    
    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        print("  No images found")
        return
    
    # Image statistics
    widths = []
    heights = []
    
    for img_file in image_files[:100]:  # Sample of 100 images
        img = cv2.imread(str(img_file))
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)
    
    print(f"  Images analyzed: {len(widths)}")
    print(f"  Average width: {np.mean(widths):.0f}px")
    print(f"  Average height: {np.mean(heights):.0f}px")
    print(f"  Smallest image: {min(widths)}x{min(heights)}")
    print(f"  Largest image: {max(widths)}x{max(heights)}")
    
    # Plot image size distribution
    plot_image_sizes(widths, heights)

def analyze_calibration(calib_dir):
    """Analyze calibration files"""
    print("\n📐 Calibration analysis:")
    
    calib_files = list(calib_dir.glob("*.txt"))
    if not calib_files:
        print("  No calibration files found")
        return
    
    # Analyze projection matrix
    focal_lengths = []
    principal_points = []
    
    for calib_file in calib_files[:10]:  # Sample of 10 files
        with open(calib_file, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    P2 = np.array([float(x) for x in line.split()[1:13]]).reshape(3, 4)
                    fx = P2[0, 0]
                    fy = P2[1, 1]
                    cx = P2[0, 2]
                    cy = P2[1, 2]
                    
                    focal_lengths.append((fx, fy))
                    principal_points.append((cx, cy))
                    break
    
    if focal_lengths:
        fx_values = [f[0] for f in focal_lengths]
        fy_values = [f[1] for f in focal_lengths]
        cx_values = [p[0] for p in principal_points]
        cy_values = [p[1] for p in principal_points]
        
        print(f"  Average focal length X: {np.mean(fx_values):.1f}")
        print(f"  Average focal length Y: {np.mean(fy_values):.1f}")
        print(f"  Average principal point X: {np.mean(cx_values):.1f}")
        print(f"  Average principal point Y: {np.mean(cy_values):.1f}")

def plot_class_distribution(class_counts):
    """Plot class distribution"""
    plt.figure(figsize=(12, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=classes, autopct='%1.1f%%')
    plt.title('Class Percentage')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_dimensions_distribution(dimensions):
    """Plot dimensions distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(dimensions['height'], bins=50, alpha=0.7)
    axes[0].set_title('Height Distribution')
    axes[0].set_xlabel('Height (meters)')
    axes[0].set_ylabel('Count')
    
    axes[1].hist(dimensions['width'], bins=50, alpha=0.7)
    axes[1].set_title('Width Distribution')
    axes[1].set_xlabel('Width (meters)')
    axes[1].set_ylabel('Count')
    
    axes[2].hist(dimensions['length'], bins=50, alpha=0.7)
    axes[2].set_title('Length Distribution')
    axes[2].set_xlabel('Length (meters)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('dimensions_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_locations_distribution(locations):
    """Plot locations distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(locations['x'], bins=50, alpha=0.7)
    axes[0].set_title('X Position Distribution')
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Count')
    
    axes[1].hist(locations['y'], bins=50, alpha=0.7)
    axes[1].set_title('Y Position Distribution')
    axes[1].set_xlabel('Y (meters)')
    axes[1].set_ylabel('Count')
    
    axes[2].hist(locations['z'], bins=50, alpha=0.7)
    axes[2].set_title('Z Position Distribution')
    axes[2].set_xlabel('Z (meters)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('locations_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_image_sizes(widths, heights):
    """Plot image size distribution"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(widths, heights, alpha=0.6)
    plt.title('Image Size Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    # Add average lines
    plt.axvline(np.mean(widths), color='red', linestyle='--', label=f'Avg Width: {np.mean(widths):.0f}')
    plt.axhline(np.mean(heights), color='blue', linestyle='--', label=f'Avg Height: {np.mean(heights):.0f}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('image_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_report():
    """Generate data analysis report"""
    print("\n📊 Generating analysis report...")
    
    report = """
# KITTI Dataset Analysis Report

## Overview
This report provides a comprehensive analysis of the KITTI dataset used for training the 3D object detection model.

## Key Statistics
- Total Images: [Calculated automatically]
- Total Objects: [Calculated automatically]
- Number of Classes: 9 main classes

## Recommendations
1. **Data Balance**: Some classes may be underrepresented
2. **Handle Truncated Objects**: High percentage of truncated objects
3. **Improve Calibration**: Use calibration files to improve accuracy

## Charts
- Class distribution
- 3D dimensions distribution
- Position distribution
- Image sizes
"""
    
    with open('data_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ Report generated: data_analysis_report.md")

if __name__ == "__main__":
    analyze_kitti_dataset()
    generate_report()