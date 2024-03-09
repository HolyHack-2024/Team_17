import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import keras_cv

from tone_model import (
    process_image,
    DEFAULT_TONE_PALETTE,
    DEFAULT_TONE_LABELS,
)
from api import process

# Settings
class_ids = ['Acne']
class_mapping = {0: 'Acne'}
model_weights_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/working/yolo_acne_detection.h5'
TEST_DATA = Path('/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/images/')
TEST_DATA_DIR = Path('/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/images/')
TEST_LABELS_FILE = TEST_DATA_DIR / 'test_labels.txt'

# Utility Functions
def load_yolov8_model(weights_path):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", include_rescaling=True)
    model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping), bounding_box_format="xyxy", backbone=backbone, fpn_depth=5)
    model.load_weights(weights_path)
    return model

def img_preprocessing(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (320, 320))
    return tf.expand_dims(img, axis=0)

def visualize_predictions_with_model(image, predictions, class_mapping):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.numpy()[0].astype("uint8"))
    if(predictions['num_detections'] <= 0):
        plt.axis('off')
        plt.show()
        return
    
    for i in range(predictions['num_detections']):
        box = predictions['boxes'][i]
        class_name = class_mapping[predictions['detection_classes'][i]]
        plt.gca().add_patch(plt.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=2, edgecolor='r', facecolor='none'))
        plt.text(box[1], box[0] - 2, f'{class_name}', bbox=dict(facecolor='red', alpha=0.5), color='white')
    plt.axis('off')
    plt.show()

def evaluate_tone_on_image(image_path, tone_palette, tone_labels):
    result = process(
        filename_or_url=str(image_path),
        image_type="auto",
        tone_palette=tone_palette,
        tone_labels=tone_labels,
        convert_to_black_white=False,
        n_dominant_colors=2,
        new_width=250,
        return_report_image=False
    )
    print(f"Image: {image_path.name}, Result: {result}")


model = load_yolov8_model(model_weights_path)

for img_path in TEST_DATA.glob('*'):
    image = img_preprocessing(str(img_path))
    predictions = model.predict(image)
    visualize_predictions_with_model(image, predictions, class_mapping)
    evaluate_tone_on_image(img_path, DEFAULT_TONE_PALETTE['color'], DEFAULT_TONE_LABELS['color'])



