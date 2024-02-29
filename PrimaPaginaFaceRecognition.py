
import cv2
import os 
import random
import numpy as np
from matplotlib import pyplot as plt

#Aplication Progrtam Interface (API)
from tensorflow.keras.models import Model
import tensorflow.keras.layers as lay

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")

cap.set(3, 176)
cap.set(4, 144)

while(True):
# Capture frame-by-frame
    ret, frame = cap.read()
# Display the resulting frame
    cv2.imshow('preview',frame)
# Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('p'):       
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()