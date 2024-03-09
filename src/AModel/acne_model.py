import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras_cv
from keras_cv import models

# Assuming class_mapping is defined globally
class_mapping = {0: 'Acne'}

def img_preprocessing(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, (320, 320))  # Resize to model's input size
    return tf.expand_dims(img, axis=0)  # Add batch dimension

def visualize_predictions_with_model(model, image, class_mapping):
    predictions = model.predict(image)
    ##check the predictions
    print(predictions)
    ##check the shape of the predictions
    print(predictions['boxes'].shape)
    print(predictions['confidence'].shape)
    print(predictions['classes'].shape)
    ##check the predictions
    print(predictions['boxes'])
    print(predictions['confidence'])
    print(predictions['classes'])
    
    
    boxes = predictions['boxes'][0]
    confidences = predictions['confidence'][0]
    class_ids = predictions['classes'][0]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image[0])

    for box, score, class_id in zip(boxes, confidences, class_ids):
        if score == -1:  # Skip invalid detections
            continue

        class_name = class_mapping.get(class_id, "Unknown")
        ymin, xmin, ymax, xmax = box

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, f'{class_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

    #plt.axis('off')
    #plt.show()

def load_yolov8_model(weights_path):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", include_rescaling=True)
    model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping), bounding_box_format="xyxy", backbone=backbone, fpn_depth=5)
    model.load_weights(weights_path)
    return model

# Define the path to your test image and model weights
test_image_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/acne.jpeg'
test_image_path2 = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/data/Acne/test/images/acne-25_jpeg.rf.61edfe1eaf01bce48403a1a22f693a86.jpg'
model_weights_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/working/yolo_acne_detection.h5'

# Load your YOLOv8 model
model = load_yolov8_model(model_weights_path)

# Preprocess the test image
preprocessed_img = img_preprocessing(test_image_path2)

# Visualize the predictions
visualize_predictions_with_model(model, preprocessed_img, class_mapping)





