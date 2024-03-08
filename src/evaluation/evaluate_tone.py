import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import logging
from typing import Union, Literal

import cv2

from src.models.base.tone_model import (
    load_image,
    is_black_white,
    DEFAULT_TONE_PALETTE,
    DEFAULT_TONE_LABELS,
    process_image,
    normalize_palette,
)
from src.extra.utils import ArgumentError

LOG = logging.getLogger(__name__)


# Path to your test dataset
TEST_DATA_DIR = Path('/path/to/your/test/data')
TEST_LABELS_FILE = TEST_DATA_DIR / 'test_labels.txt'  # A simple text file with image_path, label pairs

# Load your model - adjust this part according to how your model is saved/loaded
# For example, this could be loading a serialized model file, or initializing a model class
def load_model():
    # Placeholder: replace with your model loading code
    return None

def read_test_labels(labels_file):
    """Read test labels from a file. Expects a line format: image_path label"""
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels

def evaluate_model(test_labels, model, tone_palette, tone_labels):
    y_true, y_pred = [], []

    for img_path, true_label in test_labels.items():
        full_img_path = TEST_DATA_DIR / img_path
        image = cv2.imread(str(full_img_path))
        if image is not None:
            # Process image (e.g., detect skin tone) - adjust according to your project's function
            records, _ = process_image(
                image=image,
                is_bw=False,  # Set according to your needs or image properties
                to_bw=False,  # Set according to your needs
                skin_tone_palette=tone_palette,
                tone_labels=tone_labels,
                verbose=False  # We don't need the verbose output for evaluation
            )

            # Assuming the first record contains the prediction for the primary face
            if records:
                predicted_label = records[0]['tone_label']
                y_true.append(true_label)
                y_pred.append(predicted_label)

    # Calculate evaluation metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=tone_labels))
    print("Accuracy:", accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    model = load_model()  # Load your trained model
    test_labels = read_test_labels(TEST_LABELS_FILE)
    evaluate_model(test_labels, model, DEFAULT_TONE_PALETTE['color'], DEFAULT_TONE_LABELS['color'])
