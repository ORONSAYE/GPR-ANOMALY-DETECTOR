# main.py
import os
from inference import run_inference
from matcher import match_and_save

# === CONFIG ===
image_size = (640, 256)
base_slice_dir = r"C:\Users\oronsay\Desktop\PYTHON PROJ\bscan_combined_outputVA024"
real_model_path = r"C:\Users\oronsay\Desktop\PYTHON PROJ\REAL_MODEL\train\weights\best.pt"
migrated_model_path = r"C:\Users\oronsay\Desktop\PYTHON PROJ\MIG_MODEL\train\weights\best.pt"
output_dir = r"C:\Users\oronsay\Desktop\PYTHON PROJ\inference_outputVA024"
matched_output_dir = r"C:\Users\oronsay\Desktop\PYTHON PROJ\matched_pairsVA024"

os.makedirs(matched_output_dir, exist_ok=True)

# === PROCESS ALL SLICES ===
for slice_name in os.listdir(base_slice_dir):
    slice_path = os.path.join(base_slice_dir, slice_name)
    if not os.path.isdir(slice_path):
        continue

    real_path = os.path.join(slice_path, "real")
    migrated_path = os.path.join(slice_path, "migrated")
    if not os.path.exists(real_path) or not os.path.exists(migrated_path):
        continue

    # Inference
    run_inference(real_model_path, real_path, os.path.join(output_dir, slice_name, "real"), image_size[0])
    run_inference(migrated_model_path, migrated_path, os.path.join(output_dir, slice_name, "migrated"), image_size[0])

    # Match
    match_and_save(slice_name, image_size, output_dir, matched_output_dir)

print(f"\n[✅] All done. Matched pairs saved in: {matched_output_dir}")
