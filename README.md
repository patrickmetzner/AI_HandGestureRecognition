# AI_HandGestureRecognition
 
This project is an **image classifier** capable of distinguishing, through the computer's built in WebCam, **three different classes**: 
**1.** Positive hand gesture (thumbs up);
**2.** Negative hand gesture (thumbs down);
**3.** Absence of the previous classes (no hands).

The code to run the **image classifier** can be found in [**HandGestureRecognition.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition.py). As seen below, this program will open the computer's WebCam and store the **tree latest frames** (area inside white rectangle) captured by the camera, then the **Neural Network** ([HandGestureRecognition_UpDownNone_20k.h5](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5)) will predict the hand gestures for each frame. 

Observation: The option to analyse the latest three frames was made to minimise flickering in the displayed text.

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/HandGestureRecognition.gif" width=350> <img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/liveVideoImages.gif" width=350>


---
# About the project

This project is divided in four main files:
**1.** [**GetImages.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/GetImages.py);
    This program is responsible to gather the images that will be used to train and test the **Neural Network**. Each time the user press 's', a new batch of images will be saved with the label according to the text displayed on top of the video feed. Only the frame inside the white rectangle will be registered. Two GIF images showing the UI and the saved files can be found below.

**2.** [**NetworkTraining.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/NeuralNetworkTraining.py);
    This program will organise the images collected by the previous program and train a **Convolutional Neural Network** to recognise the three classes mentioned in the previous section. The evolution of the training after each **epoch** can be seen in the terminal window and, at the end, a **Confusion Matrix** will represent the accuracy of the predictions made by the **trained Neural Network** when taking the images in the **images/testImages** directory as input. A better explanation about the evolution of the training and testing of the [**HandGestureRecognition_UpDownNone_20k.h5**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5) **Neural Network** can be found at the end of this section. 

**3.** [**HandGestureRecognition_Test.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition_Test.py);
    This program can be used to test a **trained Neural Network**, using the images in the **images/testImages** directory as input and visualise its **Confusion Matrix** without the need of running **NetworkTraining.py** (NetworkTraining.py can take a long time to finish running).

**4.** [**HandGestureRecognition.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition.py);
    This is the program used to recognize the three classes previously mentioned. A brief explanation, together with two GIF images can be found in the first section of this file.


<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/GetImages.gif" width=350> <img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/imageFiles.gif" width=350>


For the particular case of the [**HandGestureRecognition_UpDownNone_20k.h5**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5) **Neural Network**, a set of **18.000** images where used to train the network, **3.000** images where used as validation to track the progress of each epoch (**val_loss** and ** val_accuracy**) and **3.000** images where used to test the result.

The training progress, as well as the **Confusion Matrix** can be seen below. For the specific set of images used to test the **Neural Network**, it was capable of predictiong **1.000 out of 1.000 thumbsUp** images, **998 out of 1.000 thumbsDown** images and **987 out of 1.000 noHand** images.

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/NeuralNetwork_TrainingProgress.PNG"> 

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/ConfusionMatrix.png">


---
# How to run the project

To run the project you will need **Python 3** (the project was developed using **Python 3.8.5**). To check your Python version you can run the Windows command below:
> python --version

In case you want to gather your own images and train your own **Neural Network**, you can
