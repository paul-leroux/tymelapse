# Alignment quality thresholds (in pixels or degrees)
TRANSLATION_MIN_THRESHOLD = 0  # Minimum translation (in pixels)
TRANSLATION_MAX_THRESHOLD = 100  # Maximum translation (in pixels)

SCALE_MIN_THRESHOLD_X = 0.5  # Minimum scale in x direction
SCALE_MAX_THRESHOLD_X = 1.5  # Maximum scale in x direction

SCALE_MIN_THRESHOLD_Y = 0.5  # Minimum scale in y direction
SCALE_MAX_THRESHOLD_Y = 1.5  # Maximum scale in y direction

SHEAR_MIN_THRESHOLD = 85  # Minimum shear angle (degrees, around 90)
SHEAR_MAX_THRESHOLD = 95  # Maximum shear angle (degrees, around 90)

# Feature Matching Parameters
FLANN_INDEX_KDTREE = 1  # FLANN index for KDTree
FLANN_TREES = 5  # Number of trees for FLANN
FLANN_CHECKS = 50  # Number of checks for FLANN

# Feature Detection Parameters
SIFT_RATIO = 0.7  # Ratio for good feature matches in SIFT (used in Lowe's ratio test)
AKAZE_MAX_MATCHES = 100  # Max matches to consider for AKAZE

# Other parameters
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff']  # Supported image file extensions