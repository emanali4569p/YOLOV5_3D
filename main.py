#!/usr/bin/env python3
"""
نموذج YOLOv5 محسن للكشف ثلاثي الأبعاد على مجموعة بيانات KITTI
المطور: مساعد الذكي
الوصف: نظام شامل للكشف عن الكائنات ثلاثية الأبعاد باستخدام YOLOv5
"""

import os
import sys
import time
from pathlib import Path

def print_banner():
    """طباعة شعار المشروع"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🚗 YOLOv5 للكشف ثلاثي الأبعاد على KITTI 🚗          ║
    ║                                                              ║
    ║  نظام متقدم للكشف عن الكائنات ثلاثية الأبعاد في الصور      ║
    ║  يدعم: السيارات، المشاة، الدراجات، الشاحنات وأكثر         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system():
    """فحص النظام"""
    print("🔍 فحص النظام...")
    
    # فحص Python
    python_version = sys.version_info
    print(f"  Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("  ❌ يتطلب Python 3.7 أو أحدث")
        return False
    
    # فحص PyTorch
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  CUDA: متاح - {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA: غير متاح - سيتم استخدام CPU")
    except ImportError:
        print("  ❌ PyTorch غير مثبت")
        return False
    
    # فحص البيانات
    data_dir = Path("Data")
    if not data_dir.exists():
        print("  ❌ مجلد البيانات غير موجود")
        return False
    
    required_folders = ["image_2", "label_2", "calib"]
    for folder in required_folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"  ❌ مجلد {folder} غير موجود")
            return False
        
        files = list(folder_path.glob("*"))
        if len(files) == 0:
            print(f"  ❌ مجلد {folder} فارغ")
            return False
        
        print(f"  ✅ {folder}: {len(files)} ملف")
    
    return True

def install_dependencies():
    """تثبيت المتطلبات"""
    print("\n📦 تثبيت المتطلبات...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ تم تثبيت المتطلبات بنجاح")
            return True
        else:
            print(f"  ❌ فشل في تثبيت المتطلبات: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ خطأ في تثبيت المتطلبات: {e}")
        return False

def analyze_data():
    """تحليل البيانات"""
    print("\n📊 تحليل البيانات...")
    
    try:
        from analyze_data import analyze_kitti_dataset
        analyze_kitti_dataset()
        print("  ✅ تم تحليل البيانات بنجاح")
        return True
    except Exception as e:
        print(f"  ❌ خطأ في تحليل البيانات: {e}")
        return False

def test_dataset():
    """اختبار مجموعة البيانات"""
    print("\n🧪 اختبار مجموعة البيانات...")
    
    try:
        from kitti_dataset import KITTIDataset
        
        # اختبار مجموعة التدريب
        train_dataset = KITTIDataset("Data", split="train", img_size=640)
        print(f"  مجموعة التدريب: {len(train_dataset)} عينة")
        
        # اختبار مجموعة الاختبار
        test_dataset = KITTIDataset("Data", split="test", img_size=640)
        print(f"  مجموعة الاختبار: {len(test_dataset)} عينة")
        
        # اختبار تحميل عينة
        image, targets = train_dataset[0]
        print(f"  تم تحميل عينة: {image.shape}, {len(targets)} كائن")
        
        return True
    except Exception as e:
        print(f"  ❌ خطأ في اختبار البيانات: {e}")
        return False

def test_model():
    """اختبار النموذج"""
    print("\n🤖 اختبار النموذج...")
    
    try:
        from yolo3d_model import create_model
        import torch
        
        model = create_model(nc=9)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  تم إنشاء النموذج: {total_params:,} معامل")
        
        # اختبار تمرير الأمام
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)
        
        print("  ✅ تم اختبار تمرير الأمام بنجاح")
        return True
    except Exception as e:
        print(f"  ❌ خطأ في اختبار النموذج: {e}")
        return False

def train_model():
    """تدريب النموذج"""
    print("\n🚀 بدء التدريب...")
    
    try:
        from train_3d import Trainer3D, create_config
        
        config = create_config()
        trainer = Trainer3D(config)
        
        print("  بدء التدريب...")
        trainer.train()
        
        print("  ✅ تم التدريب بنجاح")
        return True
    except Exception as e:
        print(f"  ❌ خطأ في التدريب: {e}")
        return False

def evaluate_model():
    """تقييم النموذج"""
    print("\n📊 تقييم النموذج...")
    
    try:
        from evaluate_3d import Evaluator3D
        
        checkpoint_dir = Path("checkpoints")
        best_model = checkpoint_dir / "best_model.pth"
        
        if not best_model.exists():
            print("  ❌ أفضل نموذج غير موجود")
            return False
        
        config = {'data_dir': 'Data', 'img_size': 640}
        evaluator = Evaluator3D(str(best_model), config)
        
        results = evaluator.evaluate_dataset()
        evaluator.generate_report(results, "evaluation_report.txt")
        
        print("  ✅ تم التقييم بنجاح")
        return True
    except Exception as e:
        print(f"  ❌ خطأ في التقييم: {e}")
        return False

def main():
    """الدالة الرئيسية"""
    print_banner()
    
    start_time = time.time()
    
    # فحص النظام
    if not check_system():
        print("\n❌ فشل في فحص النظام")
        return
    
    # تثبيت المتطلبات
    if not install_dependencies():
        print("\n❌ فشل في تثبيت المتطلبات")
        return
    
    # تحليل البيانات
    if not analyze_data():
        print("\n❌ فشل في تحليل البيانات")
        return
    
    # اختبار مجموعة البيانات
    if not test_dataset():
        print("\n❌ فشل في اختبار البيانات")
        return
    
    # اختبار النموذج
    if not test_model():
        print("\n❌ فشل في اختبار النموذج")
        return
    
    # تدريب النموذج
    if not train_model():
        print("\n❌ فشل في التدريب")
        return
    
    # تقييم النموذج
    if not evaluate_model():
        print("\n❌ فشل في التقييم")
        return
    
    # إنهاء ناجح
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 تم إكمال جميع المهام بنجاح!")
    print(f"⏱️  الوقت الإجمالي: {total_time/3600:.2f} ساعة")
    print(f"📁 النتائج محفوظة في:")
    print(f"   - النماذج: checkpoints/")
    print(f"   - التقارير: evaluation_report.txt")
    print(f"   - الرسوم البيانية: *.png")

if __name__ == "__main__":
    main()

