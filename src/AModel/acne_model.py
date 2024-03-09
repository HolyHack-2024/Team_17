import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras_cv
from keras_cv import models

class_mapping = {0: 'Acne'}

def visualize_predictions_with_model(model, image, class_mapping, threshold=0.5):
    """
    Visualize predictions directly from the model on a single image.

    Parameters:
    - model: The trained model that can predict bounding boxes.
    - image: The image tensor for prediction (with batch dimension).
    - class_mapping: Dictionary mapping class indices to class names.
    - threshold: Confidence threshold for displaying predictions.
    """
    predictions = model.predict(image)
    print(predictions)
    
    # Check if predictions need to be unpacked from a complex structure
    if isinstance(predictions, list):
        # Example for handling a list of dictionaries (common in TF 2.x)
        # Adjust the following lines according to your model's output format
        boxes = np.array([d['bbox'] for d in predictions])  # Example extraction
        scores = np.array([d['score'] for d in predictions])  # Example extraction
        classes = np.array([d['class'] for d in predictions]).astype(int)  # Example extraction
    elif isinstance(predictions, np.ndarray):
        # Assuming predictions are directly given as an array [ymin, xmin, ymax, xmax, score, class]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        classes = predictions[:, 5].astype(int)
    else:
        raise ValueError("Unsupported prediction format")
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image[0])
    
    for box, score, class_idx in zip(boxes, scores, classes):
        if score < threshold:
            continue
        
        class_name = class_mapping.get(class_idx, "Unknown")
        ymin, xmin, ymax, xmax = box

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, f'{class_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')
    
    plt.axis('off')
    plt.show()
    
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


# Function to preprocess the image
def preprocess_image(image_path, target_size=(320, 320)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

# Define the path to your test image and model weights
test_image_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/image_2_black.jpeg'
model_weights_path = '/Users/suleymanismaylov/Desktop/HolyHack-2024/Team_17/src/AModel/working/yolo_acne_detection.h5'

# Load your YOLOv8 model
model = load_yolov8_model(model_weights_path)

# Preprocess the test image
preprocessed_img = preprocess_image(test_image_path)

# Visualize the predictions
visualize_predictions_with_model(model, preprocessed_img, class_mapping)





