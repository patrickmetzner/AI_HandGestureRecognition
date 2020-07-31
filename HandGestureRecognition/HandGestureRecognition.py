import os
from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Global variables
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]
videoCaptureWidth = 640
videoCaptureHeight = 480
rectangleWidth = rectangleHeight = 350
imageWidth = imageHeight = 224
rectanglePoint1 = [((videoCaptureWidth - rectangleWidth) / 2), ((videoCaptureHeight - rectangleHeight) / 2)]
rectanglePoint2 = [((videoCaptureWidth + rectangleWidth) / 2), ((videoCaptureHeight + rectangleHeight) / 2)]
textSize = 1
textColor = (0, 0, 255)  # BGR color

# WebCam video capture properties
videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, videoCaptureWidth)
videoCapture.set(4, videoCaptureHeight)

myNeuralNetwork = load_model('TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5')

if os.path.isdir('liveVideo/liveVideo') is False:
    os.makedirs('liveVideo/liveVideo')

textSize = 1
textColor = (255, 255, 255)  # BGR color
numberOfFramesAnalyzed = 3
myNeuralNetwork_OutputArray = np.zeros((numberOfFramesAnalyzed,), dtype=int)
while True:
    for numberOfFrames in range(numberOfFramesAnalyzed):
        success, liveVideo = videoCapture.read()
        cv2.rectangle(liveVideo, (0, 0), (videoCaptureWidth, 40), (0, 0, 0), -1)
        cv2.rectangle(liveVideo,
                      (int(rectanglePoint1[0]), int(rectanglePoint1[1])),
                      (int(rectanglePoint2[0]), int(rectanglePoint2[1])),
                      (255, 255, 255), 2)
        image_raw = liveVideo[int(rectanglePoint1[1]):int(rectanglePoint2[1]), int(rectanglePoint1[0]):int(rectanglePoint2[0])]
        image_resized = cv2.resize(image_raw, (imageWidth, imageHeight))
        cv2.imwrite("liveVideo/liveVideo/liveVideo" + str(numberOfFrames) + ".jpg", image_resized)
        image_treated = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory='liveVideo', target_size=(imageWidth, imageHeight), batch_size=1, shuffle=False)

        myNeuralNetwork_Output = myNeuralNetwork.predict(x=image_treated, steps=1, verbose=0)
        myNeuralNetwork_OutputArray[numberOfFrames] = np.argmax(myNeuralNetwork_Output)

    liveVideo = cv2.flip(liveVideo, 1)
    imageLabelsIndex = np.mean(myNeuralNetwork_OutputArray)
    handGesture = imageLabels[int(imageLabelsIndex)]
    cv2.putText(liveVideo, handGesture, (5, 30), cv2.FONT_HERSHEY_COMPLEX, textSize, textColor, 2, cv2.LINE_AA)
    cv2.imshow("Video Capture", liveVideo)

    keyPressed = cv2.waitKey(10)
    if keyPressed == ord('q'):
        break
