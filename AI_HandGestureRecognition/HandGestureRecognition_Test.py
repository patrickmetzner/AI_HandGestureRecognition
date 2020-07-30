
numberOfImages = 50
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]
videoCaptureWidth = 640
videoCaptureHeight = 480
imageWidth = imageHeight = 224

for i in range(3):
    print(imageLabels[i])
    print(imageLabels[i] + str(i) + ".jpg")

print(100, 100)
print(500, 500)
rectanglePoint1 = [((videoCaptureWidth-imageWidth)/2), ((videoCaptureHeight-imageHeight)/2)]
print(rectanglePoint1[0], rectanglePoint1[1])
print(((videoCaptureWidth+imageWidth)/2), ((videoCaptureHeight+imageHeight)/2))

# Run next program? os.system("Image_Text.py")
