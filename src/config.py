import os


# API configuration
API_HOST="0.0.0.0"
API_PORT=5000
API_DEBUG=True

# Paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
YOLO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "yolo_weights.pt")