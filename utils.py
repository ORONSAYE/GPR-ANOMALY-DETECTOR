# utils.py
import os

def yolo_to_pixel(box, img_width, img_height):
    cls, x_c, y_c, w, h = map(float, box)
    x1 = (x_c - w / 2) * img_width
    y1 = (y_c - h / 2) * img_height
    x2 = (x_c + w / 2) * img_width
    y2 = (y_c + h / 2) * img_height
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

def read_txt_boxes(txt_path, img_width, img_height):
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [yolo_to_pixel(line.strip().split(), img_width, img_height) for line in lines]
