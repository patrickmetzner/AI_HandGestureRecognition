import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

imageWidth = imageHeight = 224
imageLabels = ["thumbsUp", "thumbsDown", "noHand"]
testPath = 'images/testImages'
validationPath = 'images/validationImages'
batchSize = 10
testBatches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=validationPath, target_size=(imageWidth, imageHeight),
                         classes=imageLabels, batch_size=batchSize, shuffle=False)

assert testBatches.num_classes == len(imageLabels)

myNeuralNetwork = load_model('TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5')

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

# Run next program? os.system("Image_Text.py")
