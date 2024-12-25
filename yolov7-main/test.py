import torch
from PIL import Image
import cv2
import numpy as np

# تحميل الموديل
def load_yolov7_model(weights_path):
    model = torch.hub.load('C:/Users/DELL/anaconda3/envs/yolo_env/yolov7-main', 'custom', weights_path, source='Local')
    model.eval()  # وضع الموديل في وضع التقييم
    return model

# معالجة الصورة وإجراء التوقعات
def predict_objects(model, image_path, conf_threshold=0.4):
    # تحميل الصورة
    img = Image.open(image_path).convert('RGB')
    
    # إجراء التوقعات
    results = model(img, size=640)  # تغيير الحجم تلقائيًا إلى 640x640
    results.print()  # طباعة النتائج في وحدة التحكم
    
    # استخراج البيانات
    predictions = results.xyxy[0].numpy()  # تحويل النتائج إلى NumPy array
    predictions = predictions[predictions[:, 4] > conf_threshold]  # تصفية النتائج بناءً على الثقة
    
    return predictions

# عرض النتائج على الصورة
def display_results(image_path, predictions, class_names):
    img = cv2.imread(image_path)
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        
        # رسم المستطيل والنص
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        num_pixels = (x2 - x1) * (y2 - y1)
        print(f"Number of pixels in bounding box {label}: {num_pixels}")
        print(f"Bounding Box for {label}:")
        print(f"Starting point (x1, y1): ({int(x1)}, {int(y1)})")
        print(f"Ending point (x2, y2): ({int(x2)}, {int(y2)})")
    # عرض الصورة
    cv2.imshow("YOLOv7 Predictions", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# المسار إلى الصورة وملف الوزن
weights_path = "C:/Users/DELL/anaconda3/envs/yolo_env/yolov7-main/yolov7.pt"  # غيّر إلى المسار الفعلي لملف الوزن
image_path = "C:/Users/DELL/anaconda3/envs/yolo_env/yolov7-main/p0.png"   # غيّر إلى المسار الفعلي للصورة

# تحميل YOLOv7
model = load_yolov7_model(weights_path)

# تنفيذ التوقعات
predictions = predict_objects(model, image_path)

# أسماء الكائنات (يمكن الحصول عليها من قائمة COCO أو موديل مخصص)
class_names = model.names  # أسماء الكائنات

# عرض النتائج
display_results(image_path, predictions, class_names)
