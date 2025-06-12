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

# Set paths
input_dir = 'input_images'
output_dir = 'output_images/pass_01'
os.makedirs(output_dir, exist_ok=True)

# Get all image paths
image_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))

# Load the reference image (first one)
ref_img = cv2.imread(image_paths[0])
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

# AKAZE detector
akaze = cv2.AKAZE_create()

# Detect features in reference
ref_kp, ref_desc = akaze.detectAndCompute(ref_gray, None)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Process each image
for idx, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect and compute keypoints/descriptors
    kp, desc = akaze.detectAndCompute(img_gray, None)

    # Match descriptors
    matches = bf.match(ref_desc, desc)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC)

    # Warp image
    aligned = cv2.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))

    # Save aligned image
    out_path = os.path.join(output_dir, f'aligned_{img_name}')
    cv2.imwrite(out_path, aligned)
    print(f'Aligned and saved: {out_path}')
