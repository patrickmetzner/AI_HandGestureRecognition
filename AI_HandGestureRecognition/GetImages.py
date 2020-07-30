import os
import cv2

numberOfImages = 50
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]
videoCaptureWidth = 640
videoCaptureHeight = 480
rectangleWidth = rectangleHeight = 350
imageWidth = imageHeight = 224

if os.path.isdir("images") is False:
    os.makedirs("images")

os.chdir('images')

videoCapture = cv2.VideoCapture(0)
videoCapture.set(3, videoCaptureWidth)
videoCapture.set(4, videoCaptureHeight)

imageCounter = 0
imageLabelCounter = 0
videoCapture_IsRunning = False
rectanglePoint1 = [((videoCaptureWidth - rectangleWidth) / 2), ((videoCaptureHeight - rectangleHeight) / 2)]
rectanglePoint2 = [((videoCaptureWidth + rectangleWidth) / 2), ((videoCaptureHeight + rectangleHeight) / 2)]
startStopMessage = "Show hand gesture: " + imageLabels[imageLabelCounter]
while True:
    keyPressed = cv2.waitKey(10)
    success, liveVideo = videoCapture.read()
    cv2.rectangle(liveVideo,
                  (int(rectanglePoint1[0]), int(rectanglePoint1[1])),
                  (int(rectanglePoint2[0]), int(rectanglePoint2[1])),
                  (255, 255, 255), 2)

    if imageCounter == numberOfImages:
        videoCapture_IsRunning = False
        imageCounter = 0
        imageLabelCounter += 1
        startStopMessage = "Change hand gesture: " + imageLabels[imageLabelCounter]
        if imageLabelCounter == (len(imageLabels)):
            break

    if videoCapture_IsRunning:
        image_raw = liveVideo[int(rectanglePoint1[1]):int(rectanglePoint2[1]),
                    int(rectanglePoint1[0]):int(rectanglePoint2[0])]
        image_resized = cv2.resize(image_raw, (imageWidth, imageHeight))
        imageName = imageLabels[imageLabelCounter] + str(imageCounter) + ".jpg"
        cv2.imwrite(imageName, image_resized)
        imageCounter += 1
        cv2.putText(liveVideo,
                    "Saving " + imageLabels[imageLabelCounter] + ": " + str(imageCounter) + " / " + str(numberOfImages),
                    (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(liveVideo,
                    startStopMessage,
                    (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if keyPressed == ord('s'):
        videoCapture_IsRunning = not videoCapture_IsRunning
        startStopMessage = "Start / Stop"

    if keyPressed == ord('q'):
        break

    cv2.imshow("Video Capture", liveVideo)

# Run next program? os.system("Image_Text.py")
