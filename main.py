# main.py
# Tymelapse orchestrator script

import cv2
from glob import glob
from config.config import *
from config.global_variables import *

from tymepkg.align import (
    detect_keypoints_and_descriptors,
    match_keypoints_flann,
    match_keypoints_bf,
    compute_homography,
    warp_image
)
from tymepkg.image_utils import (
    get_date_taken,
    preprocess_gray,
    convert_to_bgra,
    delete_previous_outputs
)
from tymepkg.log_utils import log_message
from tymepkg.metrics import compute_alignment_metrics, is_alignment_acceptable

def main():
    delete_previous_outputs(OUTPUT_DIR_PASS_01)
    delete_previous_outputs(OUTPUT_DIR_PASS_02)

    log_file_path = os.path.join(LOG_DIR, 'alignment_log.txt')
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as log_file:

        log_message(log_file, "Starting image alignment processing...\n")

        success_log = []
        fail_log = []

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

        ref_kp_sift, ref_desc_sift = detect_keypoints_and_descriptors(ref_gray, method='sift')

        for img_path in image_paths:
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

            kp, desc = detect_keypoints_and_descriptors(img_gray, method='sift')
            if desc is not None and len(kp) >= 10:
                matches = match_keypoints_flann(ref_desc_sift, desc)
                if len(matches) >= 10:
                    H = compute_homography([ref_kp_sift[m.queryIdx] for m in matches], [kp[m.trainIdx] for m in matches], matches)
                    if H is not None:
                        metrics = compute_alignment_metrics(H)
                        if is_alignment_acceptable(metrics, {
                            'translation': TRANSLATION_MAX_THRESHOLD,
                            'scale_x': SCALE_MAX_THRESHOLD_X,
                            'scale_y': SCALE_MAX_THRESHOLD_Y,
                            'shear': SHEAR_MAX_THRESHOLD
                        }):
                            aligned = warp_image(img_bgra, H, (ref_bgra.shape[1], ref_bgra.shape[0]))

            if aligned is None:
                kp_ref, desc_ref = detect_keypoints_and_descriptors(ref_gray, method='akaze')
                kp, desc = detect_keypoints_and_descriptors(img_gray, method='akaze')
                if desc_ref is not None and desc is not None and len(kp) >= 10:
                    matches = match_keypoints_bf(desc_ref, desc)[:100]
                    H = compute_homography(kp_ref, kp, matches)
                    if H is not None:
                        metrics = compute_alignment_metrics(H)
                        if is_alignment_acceptable(metrics, {
                            'translation': TRANSLATION_MAX_THRESHOLD,
                            'scale_x': SCALE_MAX_THRESHOLD_X,
                            'scale_y': SCALE_MAX_THRESHOLD_Y,
                            'shear': SHEAR_MAX_THRESHOLD
                        }):
                            aligned = warp_image(img_bgra, H, (ref_bgra.shape[1], ref_bgra.shape[0]))

            if aligned is not None:
                cv2.imwrite(out_path, aligned)
                log_message(log_file, f"Processed {img_name}: aligned and saved as {out_name}")
                success_log.append((img_name, out_name))
            else:
                log_message(log_file, f"Failed to align {img_name}: not enough good matches or homography failed.")
                fail_log.append(img_name)

        log_message(log_file, f"\nAlignment complete. Success: {len(success_log)}, Failures: {len(fail_log)}")
        if fail_log:
            log_message(log_file, "Failed images:")
            for fail in fail_log:
                log_message(log_file, f" - {fail}")

if __name__ == "__main__":
    main()
