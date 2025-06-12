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

    pts = []
    i = 0

    def onclick(event):
        nonlocal i
        if event.inaxes != ax:
            return

        if event.button == 1:
            pts.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            ax.text(event.xdata, event.ydata, str(i + 1), color='red', fontsize=12, weight='bold')
            i += 1
            plt.draw()
        elif event.button == 3 and pts:
            pts.pop()
            i -= 1
            ax.clear()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            for j, (x, y) in enumerate(pts):
                ax.plot(x, y, 'ro')
                ax.text(x, y, str(j + 1), color='red', fontsize=12, weight='bold')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    return np.array(pts, dtype=np.float32)


def main():
    os.makedirs(OUTPUT_DIR_PASS_01, exist_ok=True)
    image_list = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.jpg')])
    for i, name in enumerate(image_list):
        print(f"{i}: {name}")

    ref_index = int(input("Enter index of reference image: "))
    ref_path = os.path.join(INPUT_DIR, image_list[ref_index])
    ref_img = cv2.imread(ref_path)

    print("Select points in reference image (close window when done)...")
    pts_ref = select_points(ref_img, "Reference Image")

    # Save reference points as JSON
    import json
    ref_json_path = os.path.join(INPUT_DIR, f"meta_{os.path.splitext(image_list[ref_index])[0]}.json")
    with open(ref_json_path, 'w') as f:
        json.dump({"reference": pts_ref.tolist(), "target": None}, f, indent=2)
    print(f"Saved reference metadata to {ref_json_path}")

    import json
    for tgt_index, tgt_name in enumerate(image_list):
        if tgt_index == ref_index:
            continue
        tgt_path = os.path.join(INPUT_DIR, tgt_name)
        tgt_img = cv2.imread(tgt_path)

        print(f"Target image: {tgt_name}")

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB))
        ax.set_title("Target Image – Select Points")
        for i, (x, y) in enumerate(pts_ref):
            ax.text(x, y, str(i + 1), color='green', fontsize=12, alpha=0.4, weight='bold')
            ax.plot(x, y, 'go', alpha=0.3)

        pts_tgt = []
        i = 0

        def onclick(event):
            nonlocal i
            if event.inaxes != ax:
                return
            if event.button == 1:
                pts_tgt.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, 'bo')
                ax.text(event.xdata, event.ydata, str(i + 1), color='blue', fontsize=12, weight='bold')
                i += 1
                plt.draw()
            elif event.button == 3 and pts_tgt:
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
            plt.pause(0.1)

        fig.canvas.mpl_disconnect(cid)
        plt.close()

        pts_tgt = np.array(pts_tgt, dtype=np.float32)
        meta = {
            "reference": pts_ref.tolist(),
            "target": pts_tgt.tolist()
        }
        json_path = os.path.join(INPUT_DIR, f"meta_{os.path.splitext(tgt_name)[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata for {tgt_name} to {json_path}")
        print(f"Saved metadata for {tgt_name} to {json_path}")



    print(f"Point metadata saved to {json_path}")


if __name__ == "__main__":
    main()
