# inference.py
import os
from ultralytics import YOLO

def run_inference(model_path, image_dir, save_dir, img_width, conf=0.25):
    model = YOLO(model_path)
    model.predict(
        source=image_dir,
        save=True,
        save_txt=True,
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        exist_ok=True,
        imgsz=img_width,
        conf=conf
    )
