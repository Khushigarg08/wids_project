import cv2
import numpy as np

# Update these strings to match the names in your 'models' folder
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Load the networks using OpenCV DNN
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
MODEL_MEAN_VALUES = (78.426, 87.769, 114.896) # Standard values
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def detect_face_attributes(img_path):
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False) #
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            # Bounding box coordinates
            x1, y1 = int(detections[0, 0, i, 3] * w), int(detections[0, 0, i, 4] * h)
            x2, y2 = int(detections[0, 0, i, 5] * w), int(detections[0, 0, i, 6] * h)
            
            face = frame[max(0, y1):min(y2, h-1), max(0, x1):min(x2, w-1)]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict Gender & Age
            genderNet.setInput(face_blob)
            gender = genderList[genderNet.forward()[0].argmax()]
            
            ageNet.setInput(face_blob)
            age = ageList[ageNet.forward()[0].argmax()]
            
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Result", frame)
    cv2.waitKey(0)