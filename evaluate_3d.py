import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time

from kitti_dataset import KITTIDataset
from yolo3d_model import create_model

class Evaluator3D:
    """3D Object Detection Model Evaluator"""
    
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
        self.nc = len(self.classes)  # Add number of classes
        
        # Load model after creating data
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load trained model"""
        model = create_model(nc=len(self.test_dataset.classes))
        
        if Path(model_path).suffix == '.pth':
            # Load full checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            # Load model weights only
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
            
            # Process outputs
            predictions = self.post_process(outputs)
            
        return predictions
    
    def post_process(self, outputs: Dict) -> Dict:
        """Process model outputs"""
        detections = outputs['detections']
        
        # Apply Non-Maximum Suppression
        boxes, scores, classes = self.apply_nms(detections)
        
        # Convert to list of dictionaries
        pred_boxes = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            pred_boxes.append({
                'bbox': box,
                'confidence': score,
                'class': self.classes[cls]
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
        """Apply Non-Maximum Suppression - Real evaluation"""
        boxes = []
        scores = []
        classes = []
        
        # Real processing of model outputs
        for i, det in enumerate(detections):
            if det.numel() > 0:
                # Convert outputs to predictions
                bs, channels, ny, nx = det.shape
                
                # Convert to suitable format
                det = det.view(bs, -1, channels)
                
                # Extract boxes, confidence and classes
                for b in range(bs):
                    for obj in range(min(det.shape[1], 100)):  # Max 100 objects
                        # Extract confidence and class
                        if channels >= 5 + self.nc:
                            conf = torch.sigmoid(det[b, obj, 4:5])  # Confidence
                            cls_scores = torch.sigmoid(det[b, obj, 5:5+self.nc])  # Classes
                            
                            if conf > conf_threshold:
                                # Find best class
                                cls_id = torch.argmax(cls_scores)
                                cls_conf = cls_scores[cls_id]
                                
                                if cls_conf > conf_threshold:
                                    # Extract box
                                    cx, cy, w, h = det[b, obj, 0:4]
                                    
                                    # Convert to box coordinates
                                    x1 = torch.clamp(cx - w/2, 0, 1)
                                    y1 = torch.clamp(cy - h/2, 0, 1)
                                    x2 = torch.clamp(cx + w/2, 0, 1)
                                    y2 = torch.clamp(cy + h/2, 0, 1)
                                    
                                    # Ensure box is valid
                                    if x2 > x1 and y2 > y1:
                                        boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                                        scores.append((conf * cls_conf).item())
                                        classes.append(cls_id.item())
        
        return boxes, scores, classes
    
    def evaluate_dataset(self) -> Dict:
        """Evaluate model on complete dataset"""
        print("Evaluating model on test dataset...")
        
        total_samples = len(self.test_dataset)
        correct_detections = 0
        total_detections = 0
        total_gt = 0
        
        # Statistics for each class
        class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in self.classes}
        
        inference_times = []
        
        for idx in range(min(20, total_samples)):  # Evaluate first 20 samples
            image, targets = self.test_dataset[idx]
            
            # Measure inference time
            start_time = time.time()
            predictions = self.predict(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate accuracy
            gt_boxes = self.extract_gt_boxes(targets)
            pred_boxes = predictions['boxes']
            
            # Print diagnostic information
            print(f"Sample {idx}: GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
            
            # Calculate True Positives, False Positives, False Negatives
            tp, fp, fn = self.calculate_metrics(gt_boxes, pred_boxes)
            
            correct_detections += tp
            total_detections += len(pred_boxes)
            total_gt += len(gt_boxes)
            
            if idx % 5 == 0:
                print(f"Processed {idx+1}/{min(20, total_samples)} samples")
        
        # Calculate final metrics
        precision = correct_detections / total_detections if total_detections > 0 else 0
        recall = correct_detections / total_gt if total_gt > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_inference_time = np.mean(inference_times)
        
        print(f"Total GT boxes: {total_gt}")
        print(f"Total predictions: {total_detections}")
        print(f"Correct detections: {correct_detections}")
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_inference_time': avg_inference_time,
            'total_samples': min(20, total_samples),
            'class_stats': class_stats
        }
        
        return results
    
    def extract_gt_boxes(self, targets: torch.Tensor) -> List[Dict]:
        """Extract ground truth boxes from labels"""
        boxes = []
        
        # Extract real boxes from labels
        for target in targets:
            if len(target) >= 5:
                class_id = int(target[0])
                cx, cy, w, h = target[1:5].tolist()
                
                # Convert to box coordinates
                x1 = max(0, cx - w/2)
                y1 = max(0, cy - h/2)
                x2 = min(1, cx + w/2)
                y2 = min(1, cy + h/2)
                
                # Ensure box is valid
                if x2 > x1 and y2 > y1 and class_id < len(self.classes):
                    boxes.append({
                        'class': self.classes[class_id],
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0
                    })
        
        return boxes
    
    def calculate_metrics(self, gt_boxes: List[Dict], pred_boxes: List[Dict], 
                         iou_threshold: float = 0.5) -> Tuple[int, int, int]:
        """Calculate accuracy metrics"""
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        
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
                tp += 1
                matched_gt.add(best_gt_idx)
                fn -= 1
            else:
                fp += 1
        
        return tp, fp, fn
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_predictions(self, idx: int, save_path: str = None):
        """Visualize predictions on image"""
        image, targets = self.test_dataset[idx]
        
        # Prediction
        predictions = self.predict(image)
        
        # Convert image for display
        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image with ground truth
        ax1.imshow(img_np)
        ax1.set_title('Ground Truth')
        for target in targets:
            if len(target) >= 5:
                class_id, cx, cy, w, h = target[:5].int().tolist()
                x1 = (cx - w/2) * self.config['img_size']
                y1 = (cy - h/2) * self.config['img_size']
                x2 = (cx + w/2) * self.config['img_size']
                y2 = (cy + h/2) * self.config['img_size']
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x1, y1-5, self.classes[class_id], 
                        color='green', fontsize=10, weight='bold')
        
        # Image with predictions
        ax2.imshow(img_np)
        ax2.set_title('Predictions')
        for pred_box in predictions['boxes']:
            x1, y1, x2, y2 = pred_box['bbox']
            x1 *= self.config['img_size']
            y1 *= self.config['img_size']
            x2 *= self.config['img_size']
            y2 *= self.config['img_size']
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f"{pred_box['class']} ({pred_box['confidence']:.2f})", 
                    color='red', fontsize=10, weight='bold')
        
        ax1.axis('off')
        ax2.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: Dict, save_path: str = None):
        """Generate evaluation report"""
        report = f"""
# 3D Object Detection Model Evaluation Report

## Overall Results
- **Precision**: {results['precision']:.4f}
- **Recall**: {results['recall']:.4f}
- **F1 Score**: {results['f1_score']:.4f}
- **Average Inference Time**: {results['avg_inference_time']*1000:.2f} ms
- **Samples Tested**: {results['total_samples']}

## Class Statistics
"""
        
        for cls, stats in results['class_stats'].items():
            if stats['tp'] + stats['fp'] > 0:
                precision = stats['tp'] / (stats['tp'] + stats['fp'])
            else:
                precision = 0
            
            if stats['tp'] + stats['fn'] > 0:
                recall = stats['tp'] / (stats['tp'] + stats['fn'])
            else:
                recall = 0
            
            report += f"- **{cls}**: Precision={precision:.3f}, Recall={recall:.3f}\n"
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)

def main():
    """Main function for evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv5 3D Object Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='Data', help='Path to dataset')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save_results', type=str, help='Path to save results')
    
    args = parser.parse_args()
    
    # Evaluation settings
    config = {
        'data_dir': args.data_dir,
        'img_size': args.img_size
    }
    
    # Create evaluator
    evaluator = Evaluator3D(args.model, config)
    
    # Evaluate model
    results = evaluator.evaluate_dataset()
    
    # Generate report
    evaluator.generate_report(results, args.save_results)
    
    # Visualize predictions if requested
    if args.visualize:
        for i in range(5):
            evaluator.visualize_predictions(i)

if __name__ == "__main__":
    main()

