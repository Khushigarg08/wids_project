import cv2
import numpy as np

# 1. Update paths to look inside your 'models' folder
path = "models/"
FACE_PROTO = path + "deploy.prototxt"
FACE_MODEL = path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

AGE_PROTO = path + "age_deploy.prototxt"
AGE_MODEL = path + "age_net.caffemodel"

GENDER_PROTO = path + "gender_deploy.prototxt"
GENDER_MODEL = path + "gender_net.caffemodel"

# 2. Load the networks using OpenCV DNN
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

MODEL_MEAN_VALUES = (78.426, 87.769, 114.896) # Standard values [cite: 27]
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'] [cite: 65]
genderList = ['Male', 'Female'] [cite: 64]

# 3. New for Endterm: Initialize Webcam Capture 
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    
    # 4. Process Frame as a Blob [cite: 25]
    # Resize to 300x300 for the SSD face detector [cite: 26]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            # Calculate Bounding Box coordinates [cite: 59]
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Crop the face and create a new blob for Age/Gender prediction [cite: 60, 61]
            face = frame[max(0, y1):min(y2, h-1), max(0, x1):min(x2, w-1)]
            # Model requires 227x227 for attribute classification [cite: 60]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # 5. Predict Gender & Age [cite: 61, 63]
            genderNet.setInput(face_blob)
            gender = genderList[genderNet.forward()[0].argmax()]
            
            ageNet.setInput(face_blob)
            age = ageList[ageNet.forward()[0].argmax()]
            
            # Draw results on the frame
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Show the real-time result [cite: 69]
    cv2.imshow("Real-Time Age & Gender Guesstimator", frame)
    
    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release hardware resources
cap.release()
cv2.destroyAllWindows()