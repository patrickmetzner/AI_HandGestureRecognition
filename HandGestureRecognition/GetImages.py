import os
from cv2 import cv2

# Create directory to store images
if os.path.isdir("images") is False:
    os.makedirs("images")

os.chdir('images')

# Global variables
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]
videoCaptureWidth = 640
videoCaptureHeight = 480
rectangleWidth = rectangleHeight = 350
imageWidth = imageHeight = 224
rectanglePoint1 = [((videoCaptureWidth - rectangleWidth) / 2), ((videoCaptureHeight - rectangleHeight) / 2)]
rectanglePoint2 = [((videoCaptureWidth + rectangleWidth) / 2), ((videoCaptureHeight + rectangleHeight) / 2)]

imageCounter = imageCounterInit = 0
numberOfImages = imageCounterInit + 100
imageLabelCounter = 0
videoCapture_IsRunning = False
textSize = 1
textColor = (255, 255, 255)  # BGR color
textMessage = "Show hand gesture: " + imageLabels[imageLabelCounter]

# WebCam video capture properties
videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, videoCaptureWidth)
videoCapture.set(4, videoCaptureHeight)

while True:
    keyPressed = cv2.waitKey(10)
    if keyPressed == ord('s'):
        videoCapture_IsRunning = not videoCapture_IsRunning
        startStopMessage = "Start / Stop"

    if keyPressed == ord('q'):
        break

    success, liveVideo = videoCapture.read()
    cv2.rectangle(liveVideo, (0, 0), (videoCaptureWidth, 40), (0, 0, 0), -1)
    cv2.rectangle(liveVideo,
                  (int(rectanglePoint1[0]), int(rectanglePoint1[1])),
                  (int(rectanglePoint2[0]), int(rectanglePoint2[1])),
                  (255, 255, 255), 2)
    cv2.rectangle(liveVideo, (0, videoCaptureHeight - 40), (videoCaptureWidth, videoCaptureHeight), (0, 0, 0), -1)

    if imageCounter == numberOfImages:
        videoCapture_IsRunning = False
        imageLabelCounter += 1
        if imageLabelCounter == (len(imageLabels)):
            imageLabelCounter = 0
            imageCounterInit = imageCounter
            numberOfImages = imageCounterInit + 100
        startStopMessage = "Change hand gesture: " + imageLabels[imageLabelCounter]
        imageCounter = imageCounterInit

    if videoCapture_IsRunning:
        image_raw = liveVideo[int(rectanglePoint1[1]):int(rectanglePoint2[1]),
                    int(rectanglePoint1[0]):int(rectanglePoint2[0])]
        image_resized = cv2.resize(image_raw, (imageWidth, imageHeight))
        imageName = imageLabels[imageLabelCounter] + str(imageCounter) + ".jpg"
        cv2.imwrite(imageName, image_resized)
        imageCounter += 1
        textMessage = "Saving " + imageLabels[imageLabelCounter] + ": " + str(imageCounter) + " / " + str(numberOfImages)
    else:
        textMessage = "Show hand gesture: " + imageLabels[imageLabelCounter]

    liveVideo = cv2.flip(liveVideo, 1)
    cv2.putText(liveVideo, textMessage, (5, 30), cv2.FONT_HERSHEY_COMPLEX, textSize, textColor, 2, cv2.LINE_AA)
    cv2.putText(liveVideo, "Press 's' to start/stop & 'q' to quit",
                (5, videoCaptureHeight - 10), cv2.FONT_HERSHEY_COMPLEX, textSize, textColor, 2, cv2.LINE_AA)

    cv2.imshow("Video Capture", liveVideo)
