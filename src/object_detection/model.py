import cv2
import numpy as np

from ultralytics import YOLO
from src.config import YOLO_WEIGHTS_PATH


class TomatoLeavesDetectionModel:
    """Detect diseases in tomato leaves using YOLO model."""

    def __init__(self):
        self.model = YOLO(YOLO_WEIGHTS_PATH)
    
    def _load_img(self, image_bytes):
        """
        Preprocess the input image.

        Args:
            image_bytes (bytes): The input image.
        
        Returns:   
            Preprocessed image.
        """

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        image = cv2.resize(image, (512, 512))
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, image):
        """
        Return predictions for the input image.

        Args:
            image (bytes): The input image.
        
        Returns:
            The predicted classes, boxes, confidence, and labels.
        """

        img = self._load_img(image)
        results = self.model(img)
        boxes = results[0].boxes.cpu().numpy()
        names = results[0].names

        return {
            "classes": boxes.cls.tolist(),
            "boxes": boxes.xyxyn.tolist(),
            "conf": boxes.conf.tolist(),
            "labels": [names[i] for i in boxes.cls]
        }