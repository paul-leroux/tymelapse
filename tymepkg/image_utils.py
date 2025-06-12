import cv2
import numpy as np
import piexif

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
    b, g, r = cv2.split(img_bgr)
    alpha = np.ones_like(b) * 255
    return cv2.merge([b, g, r, alpha])

def delete_previous_outputs(output_dir):
    import os
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    os.rmdir(path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")

