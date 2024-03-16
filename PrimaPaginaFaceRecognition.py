
import cv2
import os 
import random
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from alexnet_pytorch import AlexNet
#model = AlexNet.from_pretrained('alexnet', num_classes=10)

#Aplication Progrtam Interface (API)
from tensorflow.keras.models import Model
import tensorflow.keras.layers as lay

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")

#cap.set(3, 176)
#cap.set(4, 144)
    
# Create a figure for the histogram
fig, ax = plt.subplots()
ax.set_title("Grayscale Histogram")
ax.set_xlabel("grayscale value")
ax.set_ylabel("pixel count")
ax.set_xlim([0.0, 255.0]) # Adjusted to match grayscale range


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_adjust = exposure.rescale_intensity(frame_gray, in_range=(170/255, 250/255), out_range=(0, 1))
    frame_gray_equ = cv2.equalizeHist(frame_gray)
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    ax.clear()
    ax.plot(hist)
    ax.set_xlim([0.0, 255.0])

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


    cv2.imshow('Original', frame)
    cv2.imshow('Gray', frame_gray)
    cv2.imshow('Gray', frame_gray_equ)
    cv2.imshow('Adjust', frame_gray_adjust)

    # Update the histogram plot
    plt.pause(0.01) # Pause for a short time to update the plot

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
plt.close(fig)