# *WiDS 2026: Age Guesstimator & Digit Classifier*

## *Overview*

This repository contains the end-to-end implementation of the **Age Guesstimator** project for the Women in Data Science (WiDS) 2026 program. The project is a comprehensive study of Computer Vision and Deep Learning, bridging the gap between theoretical mathematics and practical software implementation.

## *Key Features*

**MNIST Digit Classification**: A 3-layer neural network built entirely from scratch using **NumPy** to demystify backpropagation and gradient descent.

**Real-Time Inference**: Optimized webcam pipeline achieving live demographic prediction (Age and Gender) using OpenCV's DNN module.


**Manual Math Implementation**: Hand-coded forward/backward propagation, activation functions, and weight optimization.

## **Concepts Learnt**
### **1. Mathematics of Neural Networks**

**Forward Propagation**: Passed input data through weights and biases to produce predictions ().

**Activation Functions**: 
**ReLU**: , used in hidden layers to solve the vanishing gradient problem.

**Softmax**: Used in the final layer to provide probabilistic outputs for digit classification.

**Backpropagation & Optimization**: Implemented the learning loop to calculate gradients and update weights using **Gradient Descent** with an optimized learning rate () of 0.1.

### **2. Computer Vision with OpenCV**
 
**SSD Face Detection**: Leveraged a Single Shot Multibox Detector (SSD) for efficient, single-pass face localization.


**Image Blobs**: Utilized `cv2.dnn.blobFromImage` for:

**Spatial Resizing**: Scaling images to  for consistency.
 
**Mean Subtraction**: Subtracting RGB values (78.42, 87.76, 114.89) to handle illumination variations.

**Channel Swapping**: Adjusting color order from BGR to RGB.


## **Project Structure**

wids_project/
├── models/                  # Directory containing pre-trained Caffe models
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── deploy.prototxt
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   └── res10_300x300_ssd_iter_140000_fp16.caffemodel
├── age_gender_detector.py   
├── mnist_scratch.py 
├── readme.md                
## **How to Run**

### **Prerequisites**

Ensure you have Python installed and your Environment Variables (PATH) configured. Install the required libraries:

```bash
pip install numpy opencv-python get-mnist

### **Execution**

1. **Handwritten Digit Classification**:
```bash
python mnist_scratch.py

2. **Real-Time Age & Gender Detection**:
```bash
python age_gender_detector.py