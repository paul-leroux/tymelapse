# main.py
# Tymelapse (for aligning construction progress images and creating timelapse sequences)

import os
import cv2
import numpy as np
from glob import glob
import piexif
from config.config import *
from config.global_variables import *

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
    if img_bgr.shape[2] == 4:
        return img_bgr
    else:
        b, g, r = cv2.split(img_bgr)
        alpha = np.ones_like(b) * 255
        return cv2.merge([b, g, r, alpha])

def delete_previous_outputs(output_dir):
    """Delete all files in the given output directory."""
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Delete directory
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def log_message(log_file, message):
    """Helper function to write messages to the log file and flush."""
    log_file.write(message + "\n")
    log_file.flush()  # Ensure immediate write to file

def main():
    # Delete previous output files before running the alignment
    delete_previous_outputs(OUTPUT_DIR_PASS_01)
    delete_previous_outputs(OUTPUT_DIR_PASS_02)

    # Open the log file in append mode for continuous logging
    log_file_path = os.path.join(LOG_DIR, 'alignment_log.txt')
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as log_file:

        # Log the start of the processing
        log_message(log_file, "Starting image alignment processing...\n")

        # Initialize success and fail logs
        success_log = []
        fail_log = []

        # First pass alignment
        input_dir = INPUT_DIR
        output_dir_first_pass = OUTPUT_DIR_PASS_01
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
        log_message(log_file, f"Reference image selected: {os.path.basename(ref_img_path)}")

        ref_img = cv2.imread(ref_img_path)
        ref_gray = preprocess_gray(ref_img)
        ref_bgra = convert_to_bgra(ref_img)

        sift = cv2.SIFT_create()
        akaze = cv2.AKAZE_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
        search_params = dict(checks=FLANN_CHECKS)
        flann_sift = cv2.FlannBasedMatcher(index_params, search_params)
        bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        ref_kp_sift, ref_desc_sift = sift.detectAndCompute(ref_gray, None)

        # First pass processing
        for idx, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            aligned = None
            H = None

            date_taken = get_date_taken(img_path)
            out_name = f"{date_taken}.png" if date_taken else f"aligned_{img_name[:-4]}.png"
            out_path = os.path.join(output_dir_first_pass, out_name)

            print(f"Processing '{img_name}'...")

            img = cv2.imread(img_path)
            img_gray = preprocess_gray(img)
            img_bgra = convert_to_bgra(img)

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

            # After computing the metrics
            tx, ty = H[0, 2], H[1, 2]
            translation_mag = round(np.sqrt(tx ** 2 + ty ** 2), 2)
            scale_x = round(np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2), 3)
            scale_y = round(np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2), 3)
            cos_angle = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / (
                    np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2) * np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
            )
            shear_angle = round(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))), 2)

            # Check if metrics pass the threshold
            if (translation_mag > TRANSLATION_MAX_THRESHOLD or
                scale_x > SCALE_MAX_THRESHOLD_X or scale_y > SCALE_MAX_THRESHOLD_Y or
                shear_angle > SHEAR_MAX_THRESHOLD):
                log_message(log_file, f"Failed to align {img_name}: translation={translation_mag} px, scale=({scale_x}, {scale_y}), shear={shear_angle}°")
                fail_log.append(img_name)  # Add to fail_log list
            else:
                log_message(log_file, f"Processed {img_name}: aligned and saved as {out_name}")
                success_log.append((img_name, out_name))  # Add to success_log list

            # Save the aligned image
            cv2.imwrite(out_path, aligned)

        # --- Save first pass log ---
        log_message(log_file, "\nFirst pass alignment completed.\n")

        # Second pass alignment
        input_dir_first_pass = OUTPUT_DIR_PASS_01  # Output of the first pass
        output_dir_second_pass = OUTPUT_DIR_PASS_02  # New directory for second pass
        os.makedirs(output_dir_second_pass, exist_ok=True)

        image_paths = sorted(glob(os.path.join(input_dir_first_pass, '*.png')))
        success_log_second_pass = []
        fail_log_second_pass = []

        for img_path in image_paths:
            img_name = os.path.basename(img_path)

            date_taken = get_date_taken(img_path)
            out_name = f"{date_taken}_second_pass.png" if date_taken else f"aligned_{img_name[:-4]}_second_pass.png"
            out_path = os.path.join(output_dir_second_pass, out_name)

            log_message(log_file, f"Processing second pass for {img_name}...")

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image with transparency
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

            # Log the second pass results
            if aligned is not None:
                date_taken = get_date_taken(img_path)
                out_name = f"{date_taken}_second_pass.png" if date_taken else f"aligned_{img_name[:-4]}_second_pass.png"
                out_path = os.path.join(output_dir_second_pass, out_name)

                cv2.imwrite(out_path, aligned)
                log_message(log_file, f"Second pass aligned and saved: {out_path}")
                success_log_second_pass.append((img_name, out_name))
            else:
                log_message(log_file, f"Failed second pass for {img_name}")
                fail_log_second_pass.append(img_name)

        # --- Save second pass log ---
        log_message(log_file, "\nSecond pass alignment completed.\n")

if __name__ == "__main__":
    main()
