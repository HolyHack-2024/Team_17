# Comprehensive Skin Analysis and Product Recommendation System

## Project Overview

This system provides a bespoke skincare solution by evaluating user-specific skin concerns, emphasizing acne severity, skin type, and skin tone. Leveraging sophisticated machine learning models, the system offers personalized product recommendations conducive to each user's unique skincare needs.

## Structure

`TEAM_17` consists of two core components:

- `AModel` - Acne detection module using a YOLOv8-based neural network.
- `TModel` - Skin tone classification module for tailored skincare recommendations.

### Acne Detection - `AModel`

The acne detection module is capable of identifying varying levels of acne severity using state-of-the-art image processing techniques.

#### Contents:

- `checkpoints` - Contains saved states during model training.
- `save_model` - Stores the final trained model weights.
- `working` - The operational directory with the production-ready model.
- `acne_model.py` - The executable script for acne detection.
- `acne-detection-train.ipynb` - A Jupyter notebook outlining the training regimen.
- `acne.jpeg` - A test image for demonstration.

### Tone Analysis - `TModel`

This module assesses skin tone to further customize the product recommendations.

#### Contents:

- `api.py` - API interface for image processing tasks.
- `combine_predict.py` - Integrates output from both acne detection and tone analysis.
- `evaluate_tone.py` - Evaluates the tone analysis model's performance.
- `tone_model.py` - Encapsulates the tone analysis logic.
- `utils.py` - Utility functions for the module.

## Workflow

1. **Image Capture**: User's facial image is captured as the primary input.
2. **Processing**: Both `AModel` and `TModel` process the image to determine skin metrics.
3. **Results Compilation**: The outputs are aggregated and sent to the recommendation system.
4. **Additional Input**: Users can provide extra information to refine the recommendations.
5. **Recommendation System**: Generates tailored skincare product recommendations.
6. **Output**: Users receive personalized product suggestions.

## Getting Started

Clone the project repository and follow the setup instructions for each model directory.

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- OpenCV
- Matplotlib
- Jupyter (if running notebooks)

### Installation

Use `pip` to install the required packages:

```shell
pip install -r requirements.txt
