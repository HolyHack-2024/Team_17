from pathlib import Path
import logging
from typing import Union, Literal
import sys

# Add the root directory of the project to the system path
# Assuming the script is run from within the src/evaluation directory
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Import your model and any other necessary functions
from tone_model import (
    load_image,
    is_black_white,
    DEFAULT_TONE_PALETTE,
    DEFAULT_TONE_LABELS,
    process_image,
    normalize_palette,
)
from utils import ArgumentError

from api import process

LOG = logging.getLogger(__name__)


# Path to your test dataset
TEST_DATA_DIR = Path('/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/')
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

def evaluate_model(test_labels, tone_palette, tone_labels):
    y_true, y_pred = [], []

    for img_path, true_label in test_labels.items():
        full_img_path = TEST_DATA_DIR / img_path
        result = process(
            filename_or_url=str(full_img_path),
            image_type="auto",  # Let the API decide or set based on your needs
            tone_palette=tone_palette,
            tone_labels=tone_labels,
            convert_to_black_white=False,  # Set based on your needs
            n_dominant_colors=2,
            new_width=250,  # Adjust as necessary
            return_report_image=False  # We don't need the report image for evaluation
        )

        # Check if processing was successful and a face was detected
        if result.get('faces') and len(result['faces']) > 0:
            predicted_label = result['faces'][0]['tone_label']
            y_true.append(true_label)
            y_pred.append(predicted_label)

    # Calculate and print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=tone_labels))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    
if __name__ == "__main__":
    model = load_model()  # Load your trained model
    test_labels = read_test_labels(TEST_LABELS_FILE)
    evaluate_model(test_labels, DEFAULT_TONE_PALETTE['color'], DEFAULT_TONE_LABELS['color'])
