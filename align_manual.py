# align_manual.py
# Manual homography alignment by selecting corresponding points

import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Enables interactive mode for point selection
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

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Target Image – Select Points")

    # Overlay faint reference labels
    for i, (x, y) in enumerate(pts_ref):
        ax.text(x, y, str(i + 1), color='green', fontsize=12, alpha=0.4, weight='bold')
        ax.plot(x, y, 'go', alpha=0.3)

    pts_tgt = []

    def onclick(event):
        nonlocal i
        if event.inaxes != ax:
            return

        if event.button == 1:  # Left click to add point
            pts_tgt.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'bo')
            ax.text(event.xdata, event.ydata, str(i + 1), color='blue', fontsize=12, weight='bold')
            i += 1
        elif event.button == 3 and pts_tgt:  # Right click to undo
            print("Undo last point")
            pts_tgt.pop()
            i -= 1
            ax.clear()
            ax.imshow(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB))
            ax.set_title("Target Image – Select Points")
            for j, (x, y) in enumerate(pts_ref):
                ax.text(x, y, str(j + 1), color='green', fontsize=12, alpha=0.4, weight='bold')
                ax.plot(x, y, 'go', alpha=0.3)
            for j, (x, y) in enumerate(pts_tgt):
                ax.plot(x, y, 'bo')
                ax.text(x, y, str(j + 1), color='blue', fontsize=12, weight='bold')

        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    last_i = -1
    while i < len(pts_ref):
        if i != last_i:
            print(f"[STEP {i + 1}] Click on target point matching reference point {i + 1} (right-click to undo)")
            last_i = i
        plt.pause(0.1)  # Allow interaction

    fig.canvas.mpl_disconnect(cid)
    plt.show()
    plt.close()
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
