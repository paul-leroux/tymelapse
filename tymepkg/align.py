import cv2
import numpy as np

def detect_keypoints_and_descriptors(image, method='sift'):
    if method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported method: {method}")
    return detector.detectAndCompute(image, None)

def match_keypoints_flann(desc1, desc2):
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

def match_keypoints_bf(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

def compute_homography(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H

def warp_image(image, H, output_shape):
    return cv2.warpPerspective(
        image, H, output_shape,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )