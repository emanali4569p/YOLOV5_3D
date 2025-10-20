@echo off
echo ========================================
echo    نموذج YOLOv5 للكشف ثلاثي الأبعاد
echo ========================================
echo.

echo تثبيت المتطلبات...
pip install -r requirements.txt

echo.
echo اختبار النظام...
python run.py --test

echo.
echo بدء التدريب...
python run.py --train

echo.
echo تقييم النموذج...
python run.py --evaluate

echo.
echo تم الانتهاء!
pause

