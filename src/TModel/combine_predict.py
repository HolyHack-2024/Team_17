# %%
# Importing dependencies

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import *
import keras_cv
import keras_core as keras


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


class_ids = ['Acne']
class_mapping = {0: 'Acne'}
model_weights_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/working/yolo_acne_detection.h5'

TEST_DATA_DIR = Path('/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/')
TEST_LABELS_FILE = TEST_DATA_DIR / 'test_labels.txt'  # A simple text file with image_path, label pairs

BATCH_SIZE = 64
AUTO = tf.data.experimental.AUTOTUNE  # Ensure this is set for optimal parallel processing
tf.config.list_physical_devices('GPU')

# reading and resizing images
def img_preprocessing(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.cast(img, tf.float32) 
    
    return img


resizing = keras_cv.layers.JitteredResize(
    target_size=(320, 320),
    scale_factor=(0.8, 1.25),
    bounding_box_format="xyxy")

# loading dataset
def load_ds(img_paths, classes, bbox):
    img = img_preprocessing(img_paths)

    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox }
    
    return {"images": img, "bounding_boxes": bounding_boxes}

def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


# a function for converting txt file to list
def parse_txt_annot(img_path, txt_path):
    img = cv2.imread(img_path)
    
    # Check if the image was not read properly
    if img is None:
        raise FileNotFoundError(f"The image at path {img_path} was not found or cannot be read.")

    w = int(img.shape[0])
    h = int(img.shape[1])

    file_label = open(txt_path, "r")
    lines = file_label.read().split('\n')
    
    boxes = []
    classes = []
    
    if lines[0] == '':
        return img_path, classes, boxes
    else:
        for i in range(0, int(len(lines))):
            objbud=lines[i].split(' ')
            class_ = int(objbud[0])
        
            x1 = float(objbud[1])
            y1 = float(objbud[2])
            w1 = float(objbud[3])
            h1 = float(objbud[4])
        
            xmin = int((x1*w) - (w1*w)/2.0)
            ymin = int((y1*h) - (h1*h)/2.0)
            xmax = int((x1*w) + (w1*w)/2.0)
            ymax = int((y1*h) + (h1*h)/2.0)
    
            boxes.append([xmin ,ymin ,xmax ,ymax])
            classes.append(class_)
    
    return img_path, classes, boxes


# a function for creating file paths list 
def create_paths_list(path):
    full_path = []
    # Filter out files that are not images (such as .DS_Store)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".txt"}  # Add more if needed
    images = [img for img in sorted(os.listdir(path)) if img.lower().endswith(tuple(valid_extensions))]
    
    for img in images:
        full_path.append(os.path.join(path, img))
        
    return full_path

# %%
# a function for creating a dict format of files
def creating_files(img_files_paths, annot_files_paths):
    
    img_files = create_paths_list(img_files_paths)
    annot_files = create_paths_list(annot_files_paths)
    
    print(f"Found {len(img_files)} images and {len(annot_files)} annotations.")
    
    image_paths = []
    bbox = []
    classes = []
    
    for i in range(0,len(img_files)):
        image_path_, classes_, bbox_ = parse_txt_annot(img_files[i], annot_files[i])
        image_paths.append(image_path_)
        bbox.append(bbox_)
        classes.append(classes_)
        
    image_paths = tf.ragged.constant(image_paths)
    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)
    
    return image_paths, classes, bbox


def visualize_predict_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))

    y_pred = model.predict(images, verbose = 0)
    ##y_pred = keras_cv.bounding_box.to_ragged(y_pred)
    
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        true_color = (192, 57, 43),
        pred_color=(255, 235, 59),
        scale = 8,
        font_scale = 0.8,
        line_thickness=2,
        dpi = 100,
        rows=2,
        cols=2,
        show=True,
        class_mapping=class_mapping,
    )

test_img_paths, test_classes, test_bboxes = creating_files('/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/images',
                                                          '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/labels')

test_loader = tf.data.Dataset.from_tensor_slices((test_img_paths, test_classes, test_bboxes))
test_dataset = (test_loader
                .map(load_ds, num_parallel_calls = AUTO)
                .cache()
                .ragged_batch(16, drop_remainder = True)
                .map(resizing, num_parallel_calls = AUTO)
                .map(dict_to_tuple, num_parallel_calls = AUTO)
                .prefetch(AUTO))


def read_test_labels(labels_file):
    """Read test labels from a file. Expects a line format: image_path label"""
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels

def evaluate_tone(test_labels, tone_palette, tone_labels):
    y_true, y_pred = [], []

    for img_path in test_labels.items():
        full_img_path = TEST_DATA_DIR / img_path
        result = process(
            filename_or_url=str(full_img_path),
            image_type="auto",
            tone_palette=tone_palette,
            tone_labels=tone_labels,
            convert_to_black_white=False,
            n_dominant_colors=2,
            new_width=250,
            return_report_image=False
        )
        
        # Print the result for debugging
        print(f"Image: {img_path}, Result: {result}")
        
        
        

def load_yolov8_model(weights_path):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", include_rescaling=True)
    model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping), bounding_box_format="xyxy", backbone=backbone, fpn_depth=5)
    model.load_weights(weights_path)
    return model

model = load_yolov8_model(model_weights_path)

if tf.data.experimental.cardinality(test_dataset).numpy() > 0:
    test_labels = read_test_labels(TEST_LABELS_FILE)
    evaluate_tone(test_labels, DEFAULT_TONE_PALETTE['color'], DEFAULT_TONE_LABELS['color'])
    visualize_predict_detections(model, test_dataset, bounding_box_format="xyxy")
else:
    print("The dataset is empty. Please check your data loading and preprocessing steps.")

