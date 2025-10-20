#!/usr/bin/env python3
"""
سكريبت تشغيل سهل لنموذج YOLOv5 للكشف ثلاثي الأبعاد
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_requirements():
    """فحص المتطلبات"""
    print("🔍 فحص المتطلبات...")
    
    # فحص Python
    if sys.version_info < (3, 7):
        print("❌ يتطلب Python 3.7 أو أحدث")
        return False
    
    # فحص PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch غير مثبت")
        return False
    
    # فحص CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA متاح - {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA غير متاح - سيتم استخدام CPU")
    
    return True

def install_requirements():
    """تثبيت المتطلبات"""
    print("📦 تثبيت المتطلبات...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ تم تثبيت المتطلبات بنجاح")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في تثبيت المتطلبات")
        return False

def check_data():
    """فحص البيانات"""
    print("📁 فحص البيانات...")
    
    data_dir = Path("Data")
    required_dirs = ["image_2", "label_2", "calib"]
    
    if not data_dir.exists():
        print("❌ مجلد البيانات غير موجود")
        return False
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"❌ مجلد {dir_name} غير موجود")
            return False
        
        files = list(dir_path.glob("*"))
        if len(files) == 0:
            print(f"❌ مجلد {dir_name} فارغ")
            return False
        
        print(f"✅ {dir_name}: {len(files)} ملف")
    
    return True

def test_dataset():
    """اختبار مجموعة البيانات"""
    print("🧪 اختبار مجموعة البيانات...")
    
    try:
        from kitti_dataset import KITTIDataset
        
        # اختبار مجموعة التدريب
        train_dataset = KITTIDataset("Data", split="train", img_size=640)
        print(f"✅ مجموعة التدريب: {len(train_dataset)} عينة")
        
        # اختبار مجموعة الاختبار
        test_dataset = KITTIDataset("Data", split="test", img_size=640)
        print(f"✅ مجموعة الاختبار: {len(test_dataset)} عينة")
        
        # اختبار تحميل عينة
        image, targets = train_dataset[0]
        print(f"✅ تم تحميل عينة: {image.shape}, {len(targets)} كائن")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار البيانات: {e}")
        return False

def test_model():
    """اختبار النموذج"""
    print("🤖 اختبار النموذج...")
    
    try:
        from yolo3d_model import create_model
        
        model = create_model(nc=9)
        print(f"✅ تم إنشاء النموذج: {sum(p.numel() for p in model.parameters()):,} معامل")
        
        # اختبار تمرير الأمام
        import torch
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)
        
        print("✅ تم اختبار تمرير الأمام بنجاح")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار النموذج: {e}")
        return False

def start_training():
    """بدء التدريب"""
    print("🚀 بدء التدريب...")
    
    try:
        subprocess.run([sys.executable, "train_3d.py"], check=True)
        print("✅ تم التدريب بنجاح")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في التدريب")
        return False

def evaluate_model():
    """تقييم النموذج"""
    print("📊 تقييم النموذج...")
    
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("❌ مجلد النقاط المحفوظة غير موجود")
        return False
    
    best_model = checkpoint_dir / "best_model.pth"
    if not best_model.exists():
        print("❌ أفضل نموذج غير موجود")
        return False
    
    try:
        subprocess.run([sys.executable, "evaluate_3d.py", "--model", str(best_model)], 
                      check=True)
        print("✅ تم التقييم بنجاح")
        return True
    except subprocess.CalledProcessError:
        print("❌ فشل في التقييم")
        return False

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="سكريبت تشغيل نموذج YOLOv5 ثلاثي الأبعاد")
    parser.add_argument("--install", action="store_true", help="تثبيت المتطلبات")
    parser.add_argument("--test", action="store_true", help="اختبار النظام")
    parser.add_argument("--train", action="store_true", help="بدء التدريب")
    parser.add_argument("--evaluate", action="store_true", help="تقييم النموذج")
    parser.add_argument("--all", action="store_true", help="تشغيل جميع الخطوات")
    
    args = parser.parse_args()
    
    print("🎯 نموذج YOLOv5 للكشف ثلاثي الأبعاد")
    print("=" * 50)
    
    success = True
    
    # تثبيت المتطلبات
    if args.install or args.all:
        if not install_requirements():
            success = False
    
    # فحص المتطلبات
    if not check_requirements():
        success = False
    
    # فحص البيانات
    if not check_data():
        success = False
    
    # اختبار النظام
    if args.test or args.all:
        if not test_dataset():
            success = False
        if not test_model():
            success = False
    
    # التدريب
    if args.train or args.all:
        if success:
            if not start_training():
                success = False
    
    # التقييم
    if args.evaluate or args.all:
        if success:
            if not evaluate_model():
                success = False
    
    if success:
        print("\n🎉 تم إكمال جميع المهام بنجاح!")
    else:
        print("\n❌ حدثت أخطاء أثناء التنفيذ")
        sys.exit(1)

if __name__ == "__main__":
    main()

