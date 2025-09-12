# Combined update with pixel-to-GPS/depth matching
# Stores B-scans + metadata JSON for later annotation export

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
WINDOW_SIZE_METERS = 12
OVERLAP_PERCENT = 0.20
NUM_INLINES = 21
RESOLUTION_CM = 1         # crossline resolution (cm per pixel)
DEPTH_RES_M = 0.01        # depth resolution (meters per sample)

# === INPUT FILES ===
MIGRATED_ASCII_PATH = r"C:\Users\oronsay\Desktop\GPR ALGORITHM\ASCII FILES\2024-05-16-024 - Region1_3.txt"
REAL_ASCII_PATH     = r"C:\Users\oronsay\Desktop\PYTHON PROJ\ASCII FILES\2024-05-16-024 - Region1_3real.txt"
OUTPUT_BASE         = r"C:\Users\oronsay\Desktop\GPR ALGORITHM\bscan_combined_output_updatedVA024"


def parse_asci_volume(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#Volume:"):
                match = re.search(r'X-lines=(\d+).*Samples=(\d+)', line)
                if match:
                    return int(match.group(1)), int(match.group(2))
    return None, None


def load_volume_data(ascii_file_path):
    MIN_SIGNAL = -32767
    MAX_SIGNAL = 32767
    volume_data = {i: [] for i in range(1, NUM_INLINES + 1)}
    with open(ascii_file_path, 'r') as file:
        lines = [line for line in file if not line.startswith('#')]
        num_crosslines = len(lines) // NUM_INLINES

        for idx, line in enumerate(lines):
            in_line = idx // num_crosslines + 1
            values = re.split(r'[\t ]+', line.strip())  # handle tabs or spaces
            coords = (float(values[0]), float(values[1]))
            signals = [(int(val) - MIN_SIGNAL) / (MAX_SIGNAL - MIN_SIGNAL)
                       if val != '-32768' else float('nan') for val in values[2:]]
            volume_data[in_line].append({'coordinates': coords, 'signals': signals})

    return volume_data


def generate_combined_bscans(mig_volume, real_volume, samples):
    resolution_m = RESOLUTION_CM / 100
    win_crosslines = int(WINDOW_SIZE_METERS / resolution_m)
    step_crosslines = int(win_crosslines * (1 - OVERLAP_PERCENT))
    total_crosslines = len(mig_volume[1])

    depth_array = [i * DEPTH_RES_M for i in range(samples)]  # depth per row

    for start_idx in range(0, total_crosslines - win_crosslines + 1, step_crosslines):
        end_idx = start_idx + win_crosslines
        start_m = round(start_idx * resolution_m)
        end_m = round(end_idx * resolution_m)

        slice_path = os.path.join(
            OUTPUT_BASE, f"slice_{str(start_m).zfill(3)}_{str(end_m).zfill(3)}m"
        )
        mig_path = os.path.join(slice_path, "migrated")
        real_path = os.path.join(slice_path, "real")
        os.makedirs(mig_path, exist_ok=True)
        os.makedirs(real_path, exist_ok=True)

        for inline in range(1, NUM_INLINES + 1):
            mig_data = np.zeros((samples, win_crosslines))
            real_data = np.zeros((samples, win_crosslines))
            crossline_coords = []

            for i in range(win_crosslines):
                idx = start_idx + i
                if idx < len(mig_volume[inline]):
                    mig_data[:, i] = mig_volume[inline][idx]['signals']
                    real_data[:, i] = real_volume[inline][idx]['signals']
                    crossline_coords.append((
                        float(mig_volume[inline][idx]['coordinates'][0]),
                        float(mig_volume[inline][idx]['coordinates'][1])
                    ))

            # Save images
            mig_img_path = os.path.join(mig_path, f"inline_{str(inline).zfill(2)}.png")
            real_img_path = os.path.join(real_path, f"inline_{str(inline).zfill(2)}.png")
            save_bscan(mig_data, mig_img_path, cmap='jet')
            save_bscan(real_data, real_img_path, cmap='gray')

            # Save metadata
            meta = {
                "crossline_coords": crossline_coords,
                "depths": depth_array,
                "image_size": [samples, win_crosslines]  # [height, width]
            }
            with open(mig_img_path.replace(".png", ".json"), "w") as f:
                json.dump(meta, f, indent=2)
            with open(real_img_path.replace(".png", ".json"), "w") as f:
                json.dump(meta, f, indent=2)


def save_bscan(data, save_path, cmap='jet'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(data, cmap=cmap, aspect='auto', interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print("[INFO] Parsing headers...")
    _, samples_mig = parse_asci_volume(MIGRATED_ASCII_PATH)
    _, samples_real = parse_asci_volume(REAL_ASCII_PATH)
    assert samples_mig == samples_real

    print("[INFO] Loading volumes...")
    vol_mig = load_volume_data(MIGRATED_ASCII_PATH)
    vol_real = load_volume_data(REAL_ASCII_PATH)

    print("[INFO] Generating B-scans + metadata...")
    generate_combined_bscans(vol_mig, vol_real, samples_mig)
    print("[DONE] B-scans and JSON metadata saved with pixel ↔ GPS/depth mapping.")
