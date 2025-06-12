
# align_SIFT.py
# Minimal SIFT-based alignment script (no fallback, no logging)

import os
import cv2
import numpy as np
from glob import glob
from config.config import INPUT_DIR, OUTPUT_DIR_PASS_01
from tymepkg.image_utils import preprocess_gray, convert_to_bgra, get_date_taken
from tymepkg.align import detect_keypoints_and_descriptors, match_keypoints_flann, compute_homography, warp_image

def main():
    os.makedirs(OUTPUT_DIR_PASS_01, exist_ok=True)
    image_paths = sorted(glob(os.path.join(INPUT_DIR, '*.jpg')))
    if not image_paths:
        print("No input images found.")
        return

    selected = input(f"Enter index of reference image (0 to {len(image_paths) - 1}): ")
    try:
        ref_img_path = image_paths[int(selected)]
    except (IndexError, ValueError):
        print("Invalid selection. Using middle image by default.")
        ref_img_path = image_paths[len(image_paths) // 2]

    print(f"Using '{os.path.basename(ref_img_path)}' as reference image.")

    ref_img = cv2.imread(ref_img_path)
    ref_gray = preprocess_gray(ref_img)
    ref_bgra = convert_to_bgra(ref_img)
    ref_kp, ref_desc = detect_keypoints_and_descriptors(ref_gray, method='sift')

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"Processing '{img_name}'...")

        img = cv2.imread(img_path)
        img_gray = preprocess_gray(img)
        img_bgra = convert_to_bgra(img)

        kp, desc = detect_keypoints_and_descriptors(img_gray, method='sift')
        if desc is None or len(kp) < 10:
            continue

        matches = match_keypoints_flann(ref_desc, desc)
        if len(matches) < 10:
            continue

        H = compute_homography(ref_kp, kp, matches)
        if H is None:
            continue

        aligned = warp_image(img_bgra, H, (ref_bgra.shape[1], ref_bgra.shape[0]))

        date_taken = get_date_taken(img_path)
        out_name = f"{date_taken}.png" if date_taken else f"aligned_{img_name[:-4]}.png"
        out_path = os.path.join(OUTPUT_DIR_PASS_01, out_name)
        cv2.imwrite(out_path, aligned)

if __name__ == "__main__":
    main()
