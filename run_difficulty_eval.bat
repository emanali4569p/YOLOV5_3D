@echo off
echo ========================================
echo   Advanced 3D Object Detection Evaluation
echo   Difficulty-based Analysis (AP, AOS, OS)
echo ========================================
echo.

echo Running difficulty-based evaluation...
python evaluate_difficulty.py --model checkpoints/best_model.pth --data_dir Data --img_size 640 --save_results difficulty_evaluation_report.txt

echo.
echo Evaluation completed!
echo Results saved to: difficulty_evaluation_report.txt
echo.

pause
