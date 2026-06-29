# Signature Classification (CNN)

A convolutional neural network that classifies signature images, built with
TensorFlow/Keras. Includes the full pipeline from data loading and augmentation
through training, a saved model, and a desktop GUI for running predictions on
new signatures.

## What it does
- Loads signature images from per-class folders
- Applies data augmentation (random flips/rotation/zoom)
- Trains a CNN (3× Conv2D + MaxPooling blocks → Dense layers) with Keras
- Saves the trained model (`signature_model.h5`)
- Provides a desktop GUI (`gui.py`) to load an image and run inference
- A separate notebook loads the saved model and tests predictions

## Tech stack
Python · TensorFlow/Keras · OpenCV · NumPy · Pandas · Matplotlib · Jupyter

## Run locally
```bash
pip install tensorflow opencv-python numpy pandas matplotlib pillow
jupyter notebook   # open signature_detection_cnn.ipynb
```

## Status
A learning project built to practise end-to-end CNN image classification —
data pipeline, training loop, model persistence, and a simple inference UI.
The dataset is small, so it's intended as a demonstration of the workflow
rather than a production-grade model.
