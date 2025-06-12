# main.py

# Tymelapse (for aligning construction progress images and creating timelapse sequences)

# python.exe -m pip install --upgrade pip

#pip install numpy
#pip install piexif
#pip install opencv-python opencv-contrib-python
#pip install matplotlib

from config.global_variables import *

import cv2
import numpy as np
import os
from glob import glob
import piexif

def get_date_taken(img_path):
    try:
        exif_dict = piexif.load(img_path)
        date_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        # Format: "2023:05:22 12:34:56" → "2023-05-22_12-34-56"
        return date_str.replace(":", "-", 2).replace(":", "-").replace(" ", "_")
    except Exception:
        return None

# Set paths
input_dir = 'input_images'
output_dir = 'output_images/pass_01'
os.makedirs(output_dir, exist_ok=True)

# --- Preprocess: Convert to grayscale and apply CLAHE ---
def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# --- Convert image to 4-channel (BGRA) with full alpha ---
def convert_to_bgra(img_bgr):
    b, g, r = cv2.split(img_bgr)
    alpha = np.ones_like(b) * 255
    return cv2.merge([b, g, r, alpha])


# --- Get image list ---
image_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
ref_img = cv2.imread(image_paths[0])
ref_gray = preprocess_gray(ref_img)
ref_bgra = convert_to_bgra(ref_img)

# --- SIFT and AKAZE detectors ---
sift = cv2.SIFT_create()
akaze = cv2.AKAZE_create()

# --- FLANN matcher for SIFT ---
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann_sift = cv2.FlannBasedMatcher(index_params, search_params)

# --- BF matcher for AKAZE ---
bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Compute SIFT for reference ---
ref_kp_sift, ref_desc_sift = sift.detectAndCompute(ref_gray, None)

for idx, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img_gray = preprocess_gray(img)
    img_bgra = convert_to_bgra(img)

    aligned = None  # Reset for each image

    # --- Try SIFT + FLANN ---
    kp, desc = sift.detectAndCompute(img_gray, None)
    if desc is not None and len(kp) >= 10:
        matches = flann_sift.knnMatch(ref_desc_sift, desc, k=2)

        # Lowe’s ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) >= 10:
            ref_pts = np.float32([ref_kp_sift[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC)
            if H is not None:
                aligned = cv2.warpPerspective(
                    img_bgra, H, (ref_bgra.shape[1], ref_bgra.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)  # Transparent
                )

    # --- If SIFT fails, try AKAZE + BF ---
    if aligned is None:
        kp_ref, desc_ref = akaze.detectAndCompute(ref_gray, None)
        kp, desc = akaze.detectAndCompute(img_gray, None)
        if desc_ref is not None and desc is not None and len(kp) >= 10:
            matches = bf_akaze.match(desc_ref, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:100]

            ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC)
            if H is not None:
                aligned = cv2.warpPerspective(
                    img_bgra, H, (ref_bgra.shape[1], ref_bgra.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)  # Transparent
                )

    # --- Save or report failure ---
    if aligned is not None:
        #out_path = os.path.join(output_dir, f'aligned_{img_name[:-4]}.png')

        date_taken = get_date_taken(img_path)
        if date_taken:
            out_name = f"{date_taken}.png"
        else:
            out_name = f"aligned_{img_name[:-4]}.png"

        out_path = os.path.join(output_dir, out_name)


        cv2.imwrite(out_path, aligned)
        print(f"Aligned and saved: {out_path}")
    else:
        print(f"Failed to align {img_name}")