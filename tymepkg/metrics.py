import numpy as np

def compute_alignment_metrics(H):
    if H is None or H.shape != (3, 3):
        return None

    a, b, tx = H[0]
    c, d, ty = H[1]
    scale_x = np.sqrt(a**2 + b**2)
    scale_y = np.sqrt(c**2 + d**2)
    translation_mag = np.sqrt(tx**2 + ty**2)
    shear_angle = np.arctan2(a * c + b * d, a * d - b * c) * (180 / np.pi)

    return {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'translation': translation_mag,
        'shear_angle': shear_angle
    }

def is_alignment_acceptable(metrics, thresholds):
    return (
        metrics['translation'] <= thresholds['translation'] and
        metrics['scale_x'] <= thresholds['scale_x'] and
        metrics['scale_y'] <= thresholds['scale_y'] and
        abs(metrics['shear_angle']) <= thresholds['shear']
    )
