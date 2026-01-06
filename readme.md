# WiDS 2026: Age Guesstimator Project

## Overview
This repository contains the midterm submission for the Women in Data Science (WiDS) Age Guesstimator project. The project focuses on computer vision and deep learning, specifically handwritten digit classification and real-time age and gender detection.

## Project Structure
- `mnist_scratch.py`: A Neural Network built from scratch using NumPy to classify digits (0-9).
- `age_gender_detector.py`: A script using OpenCV's DNN module and pre-trained Caffe models to predict age and gender from images.
- `/models`: Folder containing `.prototxt` and `.caffemodel` files.
- `weights_W1.npy` / `weights_b1.npy`: Saved parameters from the MNIST training.

## Concepts Learnt
### 1. Mathematics of Neural Networks
Implemented forward and backward propagation manually.
- **Activation Functions**: Used ReLU for hidden layers and Softmax for the output layer.
- **Gradient Descent**: Updated weights and biases using the formula: $W = W - \alpha \cdot dW$.

### 2. Computer Vision with OpenCV
Utilized the DNN module to process images into "blobs" for model inference.
- **Face Detection**: Employed a Single Shot Multibox Detector (SSD) with ResNet.
- **Preprocessing**: Used `blobFromImage` for mean subtraction and scaling.

## How to Run
1. Install dependencies: `pip install numpy opencv-python get-mnist`
2. Run digit classification: `python mnist_scratch.py`
3. Run age detection: `python age_gender_detector.py`