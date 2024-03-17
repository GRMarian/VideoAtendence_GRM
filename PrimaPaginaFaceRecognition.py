
import cv2
import os 
import random
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams.update({'figure.max_open_warning': 0})
from alexnet_pytorch import AlexNet
#model = AlexNet.from_pretrained('alexnet', num_classes=10)

#Aplication Progrtam Interface (API)
from tensorflow.keras.models import Model
import tensorflow.keras.layers as lay

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

cap = cv2.VideoCapture(0)

program_folder_path = os.getcwd()
directory_name = "Photo_saved"
directory_path = os.path.join(program_folder_path, directory_name)

if os.path.isdir(directory_path):
    delete_files_in_directory(directory_path)
else:
    mode = 0o666
    os.mkdir(directory_path,mode)

#dir_list = os.listdir(path) REMINDER asta e ls
#os.chdir('../') REMINDER asta ma scoate din folder

crop_face = np.zeros((227,227))
if not (cap.isOpened()):
    print("Could not open video device")

#cap.set(3, 176)
#cap.set(4, 144)
    
image_index: int = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray_adjust = exposure.rescale_intensity(frame_gray, in_range=(170/255, 250/255), out_range=(0, 1))
    frame_gray_equ = cv2.equalizeHist(frame_gray)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        frame_gray_equ, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        crop_face = frame[y:y+h,x:x+w]
        crop_face = cv2.resize( crop_face, [227,227])
        type(crop_face)
        crop_nume: str = "cropPoza"+str(image_index)+".jpg"
        image_index = image_index+1
        crop_path = os.path.join(directory_path,crop_nume)
        crop_image = Image.fromarray(crop_face)
        crop_image.save(crop_path)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


    #hist, bins = np.histogram(frame_gray,256,[0, 256])
    #hist2, bins = np.histogram(frame_gray_equ,256,[0, 256])
    cv2.imshow('Original', frame)
    cv2.imshow('Gray', frame_gray)
    cv2.imshow('GrayEqu', frame_gray_equ)
    #cv2.imshow('Adjust', frame_gray_adjust)
    cv2.imshow('Crop', crop_face)
    #plt.figure("gray")
    #plt.hist(hist,bins=255,range=(0,255))
    #plt.figure("gray equ")
    #plt.hist(hist2,bins=255,range=(0,255))

    # Update the histogram plot
    #plt.pause(0.5) # Pause for a short time to update the plot

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if image_index == 10:
       break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
