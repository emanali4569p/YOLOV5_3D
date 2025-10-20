import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time
from collections import defaultdict

from kitti_dataset import KITTIDataset
from yolo3d_model import create_model

class AdvancedEvaluator3D:
    """Advanced 3D Object Detection Model Evaluator with Difficulty-based Analysis"""
    
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup dataset first
        self.test_dataset = KITTIDataset(
            data_dir=config['data_dir'],
            split='test',
            img_size=config['img_size'],
            augment=False
        )
        
        # Evaluation metrics
        self.classes = self.test_dataset.classes
        self.class_to_idx = self.test_dataset.class_to_idx
        self.nc = len(self.classes)
        
        # Load model after creating data
        self.model = self.load_model(model_path)
        
        # Difficulty thresholds based on KITTI standards
        self.difficulty_thresholds = {
            'easy': {'min_height': 40, 'max_occlusion': 0, 'max_truncation': 0.15},
            'moderate': {'min_height': 25, 'max_occlusion': 1, 'max_truncation': 0.3},
            'hard': {'min_height': 25, 'max_occlusion': 2, 'max_truncation': 0.5}
        }
        
    def load_model(self, model_path: str):
        """Load trained model"""
        model = create_model(nc=len(self.test_dataset.classes))
        
        if Path(model_path).suffix == '.pth':
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(self, image: torch.Tensor) -> Dict:
        """Predict objects in a single image"""
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            outputs = self.model(image)
            predictions = self.post_process(outputs)
        return predictions
    
    def post_process(self, outputs: Dict) -> Dict:
        """Process model outputs"""
        detections = outputs['detections']
        boxes, scores, classes = self.apply_nms(detections)
        
        # Convert to list of dictionaries
        pred_boxes = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            pred_boxes.append({
                'bbox': box,
                'confidence': score,
                'class': self.classes[cls],
                'class_id': cls
            })
        
        return {
            'boxes': pred_boxes,
            'scores': scores,
            'classes': classes,
            'depth': outputs.get('depth', []),
            'dimensions': outputs.get('dimensions', []),
            'rotation': outputs.get('rotation', [])
        }
    
    def apply_nms(self, detections, conf_threshold=0.1, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        boxes = []
        scores = []
        classes = []
        
        # NMS over raw model outputs
        
        # Process real model outputs
        for i, det in enumerate(detections):
            
            if det.numel() > 0:
                bs, channels, ny, nx = det.shape
                det = det.view(bs, -1, channels)
                
                
                for b in range(bs):
                    for obj in range(min(det.shape[1], 100)):
                        if channels >= 5 + self.nc:
                            conf = torch.sigmoid(det[b, obj, 4:5])
                            cls_scores = torch.sigmoid(det[b, obj, 5:5+self.nc])
                            
                            if conf > conf_threshold:
                                cls_id = torch.argmax(cls_scores)
                                cls_conf = cls_scores[cls_id]
                                
                                if cls_conf > conf_threshold:
                                    cx, cy, w, h = det[b, obj, 0:4]
                                    
                                    x1 = torch.clamp(cx - w/2, 0, 1)
                                    y1 = torch.clamp(cy - h/2, 0, 1)
                                    x2 = torch.clamp(cx + w/2, 0, 1)
                                    y2 = torch.clamp(cy + h/2, 0, 1)
                                    
                                    if x2 > x1 and y2 > y1:
                                        boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                                        scores.append((conf * cls_conf).item())
                                        classes.append(cls_id.item())
        
        return boxes, scores, classes
    
    def classify_difficulty(self, target: torch.Tensor) -> str:
        """Classify difficulty level based on KITTI standards"""
        if len(target) < 15:
            return 'hard'  # Default to hard if insufficient data
        
        # Extract KITTI parameters
        truncated = target[1].item()
        occluded = int(target[2].item())
        bbox_2d = target[4:8].tolist()
        
        # Calculate height in pixels (assuming image size 640)
        height_pixels = (bbox_2d[3] - bbox_2d[1]) * self.config['img_size']
        
        # Classify difficulty
        if (height_pixels >= self.difficulty_thresholds['easy']['min_height'] and
            occluded <= self.difficulty_thresholds['easy']['max_occlusion'] and
            truncated <= self.difficulty_thresholds['easy']['max_truncation']):
            return 'easy'
        elif (height_pixels >= self.difficulty_thresholds['moderate']['min_height'] and
              occluded <= self.difficulty_thresholds['moderate']['max_occlusion'] and
              truncated <= self.difficulty_thresholds['moderate']['max_truncation']):
            return 'moderate'
        else:
            return 'hard'
    
    def extract_gt_boxes_with_difficulty(self, targets: torch.Tensor) -> Dict[str, List[Dict]]:
        """Extract ground truth boxes with difficulty classification"""
        difficulty_boxes = {'easy': [], 'moderate': [], 'hard': []}
        
        # Build GT per KITTI target tensor
        
        for i, target in enumerate(targets):
            
            if len(target) >= 5:
                class_id = int(target[0])
                cx, cy, w, h = target[1:5].tolist()
                
                
                # Convert to box coordinates
                x1 = max(0, cx - w/2)
                y1 = max(0, cy - h/2)
                x2 = min(1, cx + w/2)
                y2 = min(1, cy + h/2)
                
                
                if x2 > x1 and y2 > y1 and class_id < len(self.classes):
                    difficulty = self.classify_difficulty(target)
                    
                    box_info = {
                        'bbox': [x1, y1, x2, y2],
                        'class': self.classes[class_id],
                        'class_id': class_id,
                        'confidence': 1.0,
                        'difficulty': difficulty,
                        'truncated': target[1].item(),
                        'occluded': int(target[2].item()),
                        'dimensions': target[8:11].tolist() if len(target) >= 11 else [0, 0, 0],
                        'location': target[11:14].tolist() if len(target) >= 14 else [0, 0, 0],
                        'rotation_y': target[14].item() if len(target) >= 15 else 0
                    }
                    
                    difficulty_boxes[difficulty].append(box_info)
        
        return difficulty_boxes
    
    def calculate_ap(self, gt_boxes: List[Dict], pred_boxes: List[Dict], 
                    iou_threshold: float = 0.5) -> float:
        """Calculate Average Precision (AP)"""
        if not gt_boxes or not pred_boxes:
            return 0.0
        
        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou and pred_box['class'] == gt_box['class']:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_boxes)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        return ap
    
    def calculate_aos(self, gt_boxes: List[Dict], pred_boxes: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
        """Calculate Average Orientation Similarity (AOS)"""
        if not gt_boxes or not pred_boxes:
            return 0.0
        
        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
        
        aos_scores = []
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou and pred_box['class'] == gt_box['class']:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                # Calculate orientation similarity
                gt_rotation = gt_boxes[best_gt_idx].get('rotation_y', 0)
                pred_rotation = pred_box.get('rotation_y', 0)
                
                # Calculate cosine similarity for orientation
                orientation_sim = np.cos(gt_rotation - pred_rotation)
                aos_scores.append(orientation_sim)
                matched_gt.add(best_gt_idx)
        
        return np.mean(aos_scores) if aos_scores else 0.0
    
    def calculate_os(self, gt_boxes: List[Dict], pred_boxes: List[Dict], 
                    iou_threshold: float = 0.5) -> float:
        """Calculate Orientation Similarity (OS)"""
        if not gt_boxes or not pred_boxes:
            return 0.0
        
        os_scores = []
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_box = None
            
            for gt_box in gt_boxes:
                iou = self.calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou and pred_box['class'] == gt_box['class']:
                    best_iou = iou
                    best_gt_box = gt_box
            
            if best_iou >= iou_threshold and best_gt_box:
                # Calculate orientation similarity
                gt_rotation = best_gt_box.get('rotation_y', 0)
                pred_rotation = pred_box.get('rotation_y', 0)
                
                orientation_sim = np.cos(gt_rotation - pred_rotation)
                os_scores.append(orientation_sim)
        
        return np.mean(os_scores) if os_scores else 0.0
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_by_difficulty(self) -> Dict:
        """Evaluate model by difficulty levels"""
        print("Evaluating model by difficulty levels...")
        
        # Initialize results
        results = {
            'easy': {'ap': 0, 'aos': 0, 'os': 0, 'samples': 0, 'gt_boxes': 0, 'pred_boxes': 0},
            'moderate': {'ap': 0, 'aos': 0, 'os': 0, 'samples': 0, 'gt_boxes': 0, 'pred_boxes': 0},
            'hard': {'ap': 0, 'aos': 0, 'os': 0, 'samples': 0, 'gt_boxes': 0, 'pred_boxes': 0}
        }
        
        # Collect all predictions and ground truths by difficulty
        difficulty_data = {'easy': {'gt': [], 'pred': []}, 
                          'moderate': {'gt': [], 'pred': []}, 
                          'hard': {'gt': [], 'pred': []}}
        
        total_samples = len(self.test_dataset)
        
        for idx in range(min(20, total_samples)):
            image, targets = self.test_dataset[idx]
            
            # per-sample evaluation loop
            
            # Get predictions
            predictions = self.predict(image)
            pred_boxes = predictions['boxes']
            
            # Get ground truth with difficulty classification
            gt_difficulty_boxes = self.extract_gt_boxes_with_difficulty(targets)
            
            # Add to difficulty data
            for difficulty in ['easy', 'moderate', 'hard']:
                difficulty_data[difficulty]['gt'].extend(gt_difficulty_boxes[difficulty])
                
                # Add predictions (all predictions are considered for each difficulty)
                for pred_box in pred_boxes:
                    pred_box_copy = pred_box.copy()
                    pred_box_copy['difficulty'] = difficulty
                    difficulty_data[difficulty]['pred'].append(pred_box_copy)
            
            if idx % 5 == 0:
                print(f"Processed {idx+1}/{min(20, total_samples)} samples")
        
        # Calculate metrics for each difficulty
        for difficulty in ['easy', 'moderate', 'hard']:
            gt_boxes = difficulty_data[difficulty]['gt']
            pred_boxes = difficulty_data[difficulty]['pred']
            
            results[difficulty]['samples'] = min(20, total_samples)
            results[difficulty]['gt_boxes'] = len(gt_boxes)
            results[difficulty]['pred_boxes'] = len(pred_boxes)
            
            if gt_boxes and pred_boxes:
                results[difficulty]['ap'] = self.calculate_ap(gt_boxes, pred_boxes)
                results[difficulty]['aos'] = self.calculate_aos(gt_boxes, pred_boxes)
                results[difficulty]['os'] = self.calculate_os(gt_boxes, pred_boxes)
            
            print(f"{difficulty.capitalize()}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}")
        
        return results
    
    def generate_difficulty_report(self, results: Dict, save_path: str = None):
        """Generate difficulty-based evaluation report"""
        report = f"""
# 3D Object Detection Model Evaluation Report - Difficulty Analysis

## Overall Results by Difficulty Level

### Easy Level
- **Average Precision (AP)**: {results['easy']['ap']:.4f}
- **Average Orientation Similarity (AOS)**: {results['easy']['aos']:.4f}
- **Orientation Similarity (OS)**: {results['easy']['os']:.4f}
- **Ground Truth Boxes**: {results['easy']['gt_boxes']}
- **Predicted Boxes**: {results['easy']['pred_boxes']}
- **Samples**: {results['easy']['samples']}

### Moderate Level
- **Average Precision (AP)**: {results['moderate']['ap']:.4f}
- **Average Orientation Similarity (AOS)**: {results['moderate']['aos']:.4f}
- **Orientation Similarity (OS)**: {results['moderate']['os']:.4f}
- **Ground Truth Boxes**: {results['moderate']['gt_boxes']}
- **Predicted Boxes**: {results['moderate']['pred_boxes']}
- **Samples**: {results['moderate']['samples']}

### Hard Level
- **Average Precision (AP)**: {results['hard']['ap']:.4f}
- **Average Orientation Similarity (AOS)**: {results['hard']['aos']:.4f}
- **Orientation Similarity (OS)**: {results['hard']['os']:.4f}
- **Ground Truth Boxes**: {results['hard']['gt_boxes']}
- **Predicted Boxes**: {results['hard']['pred_boxes']}
- **Samples**: {results['hard']['samples']}

## Summary
- **Overall AP**: {(results['easy']['ap'] + results['moderate']['ap'] + results['hard']['ap']) / 3:.4f}
- **Overall AOS**: {(results['easy']['aos'] + results['moderate']['aos'] + results['hard']['aos']) / 3:.4f}
- **Overall OS**: {(results['easy']['os'] + results['moderate']['os'] + results['hard']['os']) / 3:.4f}

## Difficulty Classification Criteria
- **Easy**: Height ≥ 40px, Occlusion ≤ 0, Truncation ≤ 0.15
- **Moderate**: Height ≥ 25px, Occlusion ≤ 1, Truncation ≤ 0.30
- **Hard**: Height ≥ 25px, Occlusion ≤ 2, Truncation ≤ 0.50
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

def main():
    """Main function for difficulty-based evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 3D Object Detection by Difficulty')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='Data', help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--save_results', type=str, help='Path to save results')
    
    args = parser.parse_args()
    
    # Evaluation settings
    config = {
        'data_dir': args.data_dir,
        'img_size': args.img_size
    }
    
    # Create evaluator
    evaluator = AdvancedEvaluator3D(args.model, config)
    
    # Evaluate by difficulty
    results = evaluator.evaluate_by_difficulty()
    
    # Generate report
    evaluator.generate_difficulty_report(results, args.save_results)

if __name__ == "__main__":
    main()
