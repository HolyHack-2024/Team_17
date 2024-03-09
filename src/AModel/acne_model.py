import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras_cv  # Make sure keras-cv is installed

class_mapping = {0: 'Acne'}

# Define the path to your test image and model weights
test_image_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/image_2_black.jpeg'
model_weights_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/working/yolo_acne_detection.h5'

import os

if not os.path.exists(test_image_path):
    print(f"File {test_image_path} not found.")
    exit(1)

# Load your YOLOv8 model with the custom backbone or preset
def load_yolov8_model(weights_path):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone", include_rescaling=True)
    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=5)
    model.load_weights(weights_path)
    return model

model = load_yolov8_model(model_weights_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(320, 320)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

preprocessed_img = preprocess_image(test_image_path)

# Make predictions
predictions = model.predict(preprocessed_img)


# Visualize the image and the bounding box predictions
# This is a simplified example. You'll need to adjust it according to your prediction format.
def visualize_predictions(image, predictions):
    plt.imshow(image)
    # Assuming predictions contain bounding boxes in the format [xmin, ymin, xmax, ymax]
    for box in predictions:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                          linewidth=2, edgecolor='r', facecolor='none'))
    plt.show()

# Assuming your model's predictions can be directly used or you've converted them to the right format
visualize_predictions(preprocessed_img[0], predictions)




