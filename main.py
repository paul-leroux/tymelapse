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

# Thresholds for alignment quality (adjust as needed)
TRANSLATION_MIN_THRESHOLD = 0  # Minimum acceptable translation (in pixels)
TRANSLATION_MAX_THRESHOLD = 100  # Maximum acceptable translation (in pixels)

SCALE_MIN_THRESHOLD_X = 0.5  # Minimum acceptable scale in x direction
SCALE_MAX_THRESHOLD_X = 1.5  # Maximum acceptable scale in x direction

SCALE_MIN_THRESHOLD_Y = 0.5  # Minimum acceptable scale in y direction
SCALE_MAX_THRESHOLD_Y = 1.5  # Maximum acceptable scale in y direction

SHEAR_MIN_THRESHOLD = 85  # Minimum acceptable shear angle (degrees, around 90)
SHEAR_MAX_THRESHOLD = 95  # Maximum acceptable shear angle (degrees, around 90)

def get_date_taken(img_path):
    try:
        exif_dict = piexif.load(img_path)
        date_str = exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal].decode()
        return date_str.replace(":", "-", 2).replace(":", "-").replace(" ", "_")
    except Exception:
        return None

def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def convert_to_bgra(img_bgr):
    # Check if the image already has 4 channels (RGBA)
    if img_bgr.shape[2] == 4:
        return img_bgr  # Return the image as is if it's already in BGRA format
    else:
        b, g, r = cv2.split(img_bgr)
        alpha = np.ones_like(b) * 255  # Create an alpha channel with full opacity
        return cv2.merge([b, g, r, alpha])  # Merge BGR channels with the alpha channel

def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def main():
    # First pass alignment
    input_dir = 'input_images'
    output_dir_first_pass = 'output_images/pass_01'
    os.makedirs(output_dir_first_pass, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
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

    sift = cv2.SIFT_create()
    akaze = cv2.AKAZE_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_sift = cv2.FlannBasedMatcher(index_params, search_params)
    bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ref_kp_sift, ref_desc_sift = sift.detectAndCompute(ref_gray, None)

    success_log = []
    fail_log = []
    metrics_log = []  # Track alignment quality metrics

    # First pass processing
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        if img_path == ref_img_path:
            print(f"Skipping reference image: {img_name}")
            continue

        img = cv2.imread(img_path)
        img_gray = preprocess_gray(img)
        img_bgra = convert_to_bgra(img)

        aligned = None
        H = None

        kp, desc = sift.detectAndCompute(img_gray, None)
        if desc is not None and len(kp) >= 10:
            matches = flann_sift.knnMatch(ref_desc_sift, desc, k=2)
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
                        borderValue=(0, 0, 0, 0)
                    )

        # If first pass alignment failed, try AKAZE
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
                        borderValue=(0, 0, 0, 0)
                    )

        if aligned is not None:
            date_taken = get_date_taken(img_path)
            out_name = f"{date_taken}.png" if date_taken else f"aligned_{img_name[:-4]}.png"
            out_path = os.path.join(output_dir_first_pass, out_name)

            # Compute alignment metrics from H
            tx, ty = H[0, 2], H[1, 2]
            translation_mag = round(np.sqrt(tx ** 2 + ty ** 2), 2)
            scale_x = round(np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2), 3)
            scale_y = round(np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2), 3)
            cos_angle = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / (
                    np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2) * np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
            )
            shear_angle = round(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))), 2)

            metrics_log.append({
                "image": img_name,
                "output": out_name,
                "translation": translation_mag,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "shear_angle": shear_angle
            })

            cv2.imwrite(out_path, aligned)
            print(f"First pass aligned and saved: {out_path}")
            success_log.append((img_name, out_name))
        else:
            print(f"Failed first pass for {img_name}")
            fail_log.append(img_name)

    # --- Save first pass log ---
    with open("first_pass_alignment_log.txt", "w", encoding="utf-8") as log:
        log.write("Successfully aligned:\n")
        for original, saved in success_log:
            log.write(f"{original} -> {saved}\n")
        log.write("\nFailed to align:\n")
        for name in fail_log:
            log.write(f"{name}\n")
        log.write("\nAlignment metrics:\n")
        for m in metrics_log:
            log.write(f"{m['image']} -> {m['output']}: "
                      f"translation={m['translation']} px, "
                      f"scale=({m['scale_x']}, {m['scale_y']}), "
                      f"shear={m['shear_angle']}°\n")

    # Second pass alignment
    input_dir_first_pass = 'output_images/pass_01  # Output of the first pass
    output_dir_second_pass = 'output_images/pass_02' # New directory for second pass
    os.makedirs(output_dir_second_pass, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_dir_first_pass, '*.png')))

    # Loop through the aligned images from the first pass and process them again
    for img_path in image_paths:
        img_name = os.path.basename(img_path)

        print(f"Processing second pass for {img_name}...")

        img = cv2.imread(img_path)
        img_gray = preprocess_gray(img)
        img_bgra = convert_to_bgra(img)

        # Same process as before for the second pass
        aligned = None
        H = None
        kp, desc = sift.detectAndCompute(img_gray, None)
        if desc is not None and len(kp) >= 10:
            matches = flann_sift.knnMatch(ref_desc_sift, desc, k=2)
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
                        borderValue=(0, 0, 0, 0)
                    )

        # If second pass alignment failed, try AKAZE
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
                        borderValue=(0, 0, 0, 0)
                    )

        # Save the aligned image from the second pass to the new output directory
        if aligned is not None:
            date_taken = get_date_taken(img_path)
            out_name = f"{date_taken}_second_pass.png" if date_taken else f"aligned_{img_name[:-4]}_second_pass.png"
            out_path = os.path.join(output_dir_second_pass, out_name)

            cv2.imwrite(out_path, aligned)
            print(f"Second pass aligned and saved: {out_path}")
        else:
            print(f"Failed second pass for {img_name}")

if __name__ == "__main__":
    main()
