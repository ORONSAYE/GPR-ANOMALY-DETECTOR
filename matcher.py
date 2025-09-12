# matcher.py
import os
import shutil
from pathlib import Path
from utils import read_txt_boxes, iou, center_distance

def boxes_match(real_boxes, migrated_boxes, iou_thresh=0.1, center_thresh=20):
    for rb in real_boxes:
        for mb in migrated_boxes:
            if iou(rb, mb) > iou_thresh or center_distance(rb, mb) < center_thresh:
                return True
    return False

def match_and_save(slice_name, image_size, output_dir, matched_output_dir):
    slice_real_dir = os.path.join(output_dir, slice_name, "real")
    slice_migrated_dir = os.path.join(output_dir, slice_name, "migrated")
    real_labels_dir = os.path.join(slice_real_dir, "labels")
    migrated_labels_dir = os.path.join(slice_migrated_dir, "labels")

    for img_file in os.listdir(slice_real_dir):
        if not img_file.endswith(".jpg"):
            continue

        inline_num = int(Path(img_file).stem.split("_")[-1])
        real_txt = os.path.join(real_labels_dir, f"inline_{inline_num:02d}.txt")
        real_boxes = read_txt_boxes(real_txt, *image_size)

        if not real_boxes:
            continue

        for offset in [-1, 0, 1]:
            compare_num = inline_num + offset
            migrated_txt = os.path.join(migrated_labels_dir, f"inline_{compare_num:02d}.txt")
            migrated_boxes = read_txt_boxes(migrated_txt, *image_size)

            if boxes_match(real_boxes, migrated_boxes):
                real_img_path = os.path.join(slice_real_dir, f"inline_{inline_num:02d}.jpg")
                migrated_img_path = os.path.join(slice_migrated_dir, f"inline_{compare_num:02d}.jpg")

                if os.path.exists(real_img_path) and os.path.exists(migrated_img_path):
                    shutil.copy2(real_img_path, os.path.join(matched_output_dir, f"{slice_name}_real_inline_{inline_num:02d}.jpg"))
                    shutil.copy2(migrated_img_path, os.path.join(matched_output_dir, f"{slice_name}_migrated_inline_{compare_num:02d}.jpg"))
                    print(f"[✅] Match: inline_{inline_num:02d}.jpg vs inline_{compare_num:02d}.jpg (slice {slice_name})")
                else:
                    print(f"[⚠️] Missing image for inline {inline_num:02d} or {compare_num:02d}")
