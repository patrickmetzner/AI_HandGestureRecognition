import os
import random
import glob
import shutil
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Global variables
numberOfTrainingImages = 18000      # Must be multiple of len(imageLabels)
numberOfValidationImages = 3000     # Must be multiple of len(imageLabels)
numberOfTestImages = 1050           # Must be multiple of len(imageLabels)
imageWidth = imageHeight = 224
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]

# Organize images into trainingImages, validationImages, testImages directories
os.chdir('images')
if os.path.isdir('trainingImages/thumbsUp') is False:
    # Make trainingImages directory
    os.makedirs('trainingImages/thumbsUp')
    os.makedirs('trainingImages/thumbsDown')
    os.makedirs('trainingImages/noHand')

    # Make validationImages directory
    os.makedirs('validationImages/thumbsUp')
    os.makedirs('validationImages/thumbsDown')
    os.makedirs('validationImages/noHand')

    # Make testImages directory
    os.makedirs('testImages/thumbsUp')
    os.makedirs('testImages/thumbsDown')
    os.makedirs('testImages/noHand')

    # Transfer trainingImages to directory
    for c in random.sample(glob.glob('thumbsUp*'), int(numberOfTrainingImages/len(imageLabels))):
        shutil.move(c, 'trainingImages/thumbsUp')
    for c in random.sample(glob.glob('thumbsDown*'), int(numberOfTrainingImages/len(imageLabels))):
        shutil.move(c, 'trainingImages/thumbsDown')
    for c in random.sample(glob.glob('noHand*'), int(numberOfTrainingImages/len(imageLabels))):
        shutil.move(c, 'trainingImages/noHand')

    # Transfer validationImages to directory
    for c in random.sample(glob.glob('thumbsUp*'), int(numberOfValidationImages/len(imageLabels))):
        shutil.move(c, 'validationImages/thumbsUp')
    for c in random.sample(glob.glob('thumbsDown*'), int(numberOfValidationImages/len(imageLabels))):
        shutil.move(c, 'validationImages/thumbsDown')
    for c in random.sample(glob.glob('noHand*'), int(numberOfValidationImages/len(imageLabels))):
        shutil.move(c, 'validationImages/noHand')

    # Transfer testImages to directory
    for c in random.sample(glob.glob('thumbsUp*'), int(numberOfTestImages/len(imageLabels))):
        shutil.move(c, 'testImages/thumbsUp')
    for c in random.sample(glob.glob('thumbsDown*'), int(numberOfTestImages/len(imageLabels))):
        shutil.move(c, 'testImages/thumbsDown')
    for c in random.sample(glob.glob('noHand*'), int(numberOfTestImages/len(imageLabels))):
        shutil.move(c, 'testImages/noHand')

os.chdir('../')

trainingPath = 'images/trainingImages'
validationPath = 'images/validationImages'
testPath = 'images/testImages'

# Image preparation
batchSize = 10
trainingBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=trainingPath, target_size=(imageWidth, imageHeight),
                         classes=imageLabels, batch_size=batchSize)

validationBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=validationPath, target_size=(imageWidth, imageHeight),
                         classes=imageLabels, batch_size=batchSize)

testBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=testPath, target_size=(imageWidth, imageHeight),
                         classes=imageLabels, batch_size=batchSize, shuffle=False)

assert trainingBatches.n == numberOfTrainingImages
assert validationBatches.n == numberOfValidationImages
assert testBatches.n == numberOfTestImages
assert trainingBatches.num_classes == validationBatches.num_classes == testBatches.num_classes == len(imageLabels)

images, labels = next(trainingBatches)


# Function to plot images in 1x10 grid to better visualize the data
def plotImages(imagesArray):
    figure, axes = plt.subplots(1, 10, figsize=(10, 2))
    axes = axes.flatten()
    for img, ax in zip(imagesArray, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(images)
# print(labels)


myNeuralNetwork = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(imageWidth, imageHeight, 3)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=len(imageLabels), activation='softmax'), ])

print("myNeuralNetwork created.")
myNeuralNetwork.summary()

myNeuralNetwork.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
print("myNeuralNetwork compiled.")

myNeuralNetwork.fit(x=trainingBatches,
                    steps_per_epoch=len(trainingBatches),
                    validation_data=validationBatches,
                    validation_steps=len(validationBatches),
                    epochs=10,
                    verbose=2)
print("myNeuralNetwork trained.")

testImages, testLabels = next(testBatches)
# plotImages(testImages)
# print(testLabels)

predictions = myNeuralNetwork.predict(x=testBatches, steps=len(testBatches), verbose=0)

myConfusionMatrix = confusion_matrix(y_true=testBatches.classes, y_pred=np.argmax(predictions, axis=-1))


# Function to print and plot the confusion matrix. Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm=myConfusionMatrix, classes=imageLabels, title='Confusion Matrix')

if os.path.isdir('TrainedNeuralNetworks') is False:
    os.makedirs('TrainedNeuralNetworks')

myNeuralNetwork.save('TrainedNeuralNetworks/HandGestureRecognition_New.h5')

# Run next program? os.system("Image_Text.py")
