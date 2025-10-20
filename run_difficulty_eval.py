#!/usr/bin/env python3
"""
Advanced 3D Object Detection Evaluation by Difficulty Levels
Calculates AP, AOS, and OS metrics for Easy, Moderate, and Hard difficulty levels
"""

import subprocess
import sys
import os

def run_difficulty_evaluation():
    """Run difficulty-based evaluation"""
    print("🚀 Starting Advanced 3D Object Detection Evaluation")
    print("=" * 60)
    
    # Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using: python train_3d.py")
        return False
    
    # Run evaluation
    try:
        cmd = [
            sys.executable, "evaluate_difficulty.py",
            "--model", model_path,
            "--data_dir", "Data",
            "--img_size", "640",
            "--save_results", "difficulty_evaluation_report.txt"
        ]
        
        print("📊 Running difficulty-based evaluation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Evaluation completed successfully!")
        print("\n📋 Results saved to: difficulty_evaluation_report.txt")
        
        # Show summary
        if os.path.exists("difficulty_evaluation_report.txt"):
            with open("difficulty_evaluation_report.txt", "r", encoding="utf-8") as f:
                content = f.read()
                print("\n" + "="*60)
                print("EVALUATION SUMMARY")
                print("="*60)
                
                # Extract key metrics
                lines = content.split('\n')
                for line in lines:
                    if 'Average Precision' in line or 'Average Orientation' in line or 'Overall' in line:
                        print(line)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🎯 Advanced 3D Object Detection Evaluation")
    print("Evaluates model performance by difficulty levels:")
    print("  • Easy: Height ≥ 40px, Occlusion ≤ 0, Truncation ≤ 0.15")
    print("  • Moderate: Height ≥ 25px, Occlusion ≤ 1, Truncation ≤ 0.30")
    print("  • Hard: Height ≥ 25px, Occlusion ≤ 2, Truncation ≤ 0.50")
    print()
    
    success = run_difficulty_evaluation()
    
    if success:
        print("\n🎉 Evaluation completed successfully!")
        print("📁 Check 'difficulty_evaluation_report.txt' for detailed results")
    else:
        print("\n❌ Evaluation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
