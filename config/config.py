import os

# Paths for directories
INPUT_DIR = os.path.join(os.getcwd(), 'data', 'input_images')  # Path for input images
OUTPUT_DIR_PASS_01 = os.path.join(os.getcwd(), 'data', 'output_images', 'pass_01')  # First pass output
OUTPUT_DIR_PASS_02 = os.path.join(os.getcwd(), 'data', 'output_images', 'pass_02')  # Second pass output
LOG_DIR = os.path.join(os.getcwd(), 'logs')  # Directory for storing logs

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_PASS_01, exist_ok=True)
os.makedirs(OUTPUT_DIR_PASS_02, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging and error logs
LOG_FILE = os.path.join(LOG_DIR, 'alignment_log.txt')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'error_log.txt')