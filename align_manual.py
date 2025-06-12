# align_manual.py
# Manual homography alignment by selecting corresponding points

import os
import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Enable interactive mode in PyCharm

import matplotlib.pyplot as plt
from config.config import INPUT_DIR, OUTPUT_DIR_PASS_01

def select_points(img, title):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    pts = plt.ginput(n=-1, timeout=0)

    # Annotate selected points with numbers
    for i, (x, y) in enumerate(pts):
        ax.text(x, y, str(i + 1), color='red', fontsize=12, weight='bold')
        ax.plot(x, y, 'ro')

    plt.show()
    plt.close()
    return np.array(pts, dtype=np.float32)


def main():
    os.makedirs(OUTPUT_DIR_PASS_01, exist_ok=True)

    # Get reference and target image paths
    image_list = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.jpg')])
    for i, name in enumerate(image_list):
        print(f"{i}: {name}")

    ref_index = int(input("Enter index of reference image: "))
    tgt_index = int(input("Enter index of target image: "))

    ref_path = os.path.join(INPUT_DIR, image_list[ref_index])
    tgt_path = os.path.join(INPUT_DIR, image_list[tgt_index])

    ref_img = cv2.imread(ref_path)
    tgt_img = cv2.imread(tgt_path)

    print("Select points in reference image (close window when done)...")
    pts_ref = select_points(ref_img, "Reference Image")

    print("Select corresponding points in target image (same order, close window when done)...")

    # Show reference image with numbered points again for guidance
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Reference (Guidance)")
    for i, (x, y) in enumerate(pts_ref):
        ax.text(x, y, str(i + 1), color='green', fontsize=12, weight='bold')
        ax.plot(x, y, 'go')
    plt.show()

    fig, (ax_ref, ax_tgt) = plt.subplots(1, 2, figsize=(20, 9))
    ax_ref.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    ax_ref.set_title("Reference Image")
    for i, (x, y) in enumerate(pts_ref):
        ax_ref.text(x, y, str(i + 1), color='green', fontsize=12, weight='bold')
        ax_ref.plot(x, y, 'go')

    ax_tgt.imshow(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB))
    ax_tgt.set_title("Target Image – Select Points")
    ax.set_title("Target Image – Select Points (Match Reference Numbers)")

    pts_tgt = []
    for i in range(len(pts_ref)):
        print(f"[STEP {i + 1}] Click on target point matching reference point {i + 1}")
        pt = plt.ginput(n=1, timeout=0)[0]
        ax_tgt.plot(pt[0], pt[1], 'bo')
        ax_tgt.text(pt[0], pt[1], str(i + 1), color='blue', fontsize=12, weight='bold')
        pts_tgt.append(pt)
        plt.draw()

    plt.show()
    plt.close()
    pts_tgt = np.array(pts_tgt, dtype=np.float32)

    if len(pts_ref) < 4 or len(pts_tgt) < 4 or len(pts_ref) != len(pts_tgt):
        print("Need at least 4 matching points and same number of points in both images.")
        return

    H, _ = cv2.findHomography(pts_tgt, pts_ref, method=0)
    if H is None:
        print("Homography computation failed.")
        return

    aligned = cv2.warpPerspective(tgt_img, H, (ref_img.shape[1], ref_img.shape[0]))
    out_name = f"aligned_{os.path.splitext(image_list[tgt_index])[0]}.png"
    out_path = os.path.join(OUTPUT_DIR_PASS_01, out_name)

    cv2.imwrite(out_path, aligned)

    # Also save the reference image next to the aligned one
    ref_out_name = f"reference_{os.path.splitext(image_list[ref_index])[0]}.png"
    ref_out_path = os.path.join(OUTPUT_DIR_PASS_01, ref_out_name)
    cv2.imwrite(ref_out_path, ref_img)
    print(f"Aligned image saved to {out_path}")

    # Save point metadata for later use
    metadata_path = os.path.join(OUTPUT_DIR_PASS_01, f"meta_{os.path.splitext(image_list[tgt_index])[0]}.npz")
    np.savez(metadata_path, reference=pts_ref, target=pts_tgt)
    print(f"Point metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
