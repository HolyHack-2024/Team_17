Project Title: Comprehensive Skin Analysis and Product Recommendation System

Overview

Our project aims to provide a personalized skincare experience by analyzing the user's skin concerns, specifically focusing on acne level, skin type, and skin tone. Based on the analysis, our system recommends products that are tailored to the user's specific needs.

Project Structure

The project is divided into two main components, housed under the TEAM_17 directory:

AModel: Contains the acne detection model and its associated files.
TModel: Houses the tone analysis model and related scripts.
Acne Detection (AModel)

The acne detection model utilizes a pre-trained YOLOv8 network to identify and analyze acne on the user's skin.

Directory Contents
checkpoints: Stores intermediate model weights during training.
save_model: Contains saved model weights post-training.
working: Holds the final model used for predictions.
acne_model.py: The main script to run the acne detection model.
acne-detection-train.ipynb: Jupyter notebook detailing the training process.
acne.jpeg: Sample image for testing and demonstration purposes.
Tone Analysis (TModel)

The tone analysis model assesses the user's skin tone and provides a classification used to further personalize the product recommendations.

Directory Contents
api.py: The interface for interacting with the model to process images.
combine_predict.py: Script that combines predictions from both acne and tone models.
evaluate_tone.py: Used for evaluating the performance of the tone analysis model.
tone_model.py: Contains the logic for the tone analysis.
utils.py: Helper functions used across the tone analysis component.

Workflow

The workflow depicted in the architecture diagram showcases the process from image capture to product recommendation.

Capture User Image: The user's image is captured and used as input for the models.
Models: The AModel and TModel concurrently process the image to determine acne concern level, skin type, and skin tone.
Output from the Models: The results are compiled and sent to the recommendation system.
User Additional Input: If necessary, additional user input can be provided to refine the results.
Recommendation System: Based on the analysis, the system generates a set of personalized product recommendations.
Recommended Products: The user is presented with the product recommendations.
