# AI_HandGestureRecognition
 
This project is an **image classifier** capable of distinguishing, through the computer's built in WebCam, **three different classes**: 

**1.** Positive hand gesture (thumbs up).

**2.** Negative hand gesture (thumbs down).

**3.** Absence of the previous classes (no hands).


The code to run the **image classifier** can be found in [**HandGestureRecognition.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition.py). 

As seen below, this program will open the computer's WebCam and store the **tree latest frames** (area inside white rectangle) captured by the camera, then the **Neural Network** ([HandGestureRecognition_UpDownNone_20k.h5](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5)) will predict the hand gestures for each frame. 

Observation: The option to analyse the latest three frames was made to minimise flickering in the displayed text.

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/HandGestureRecognition.gif" width=350> <img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/liveVideoImages.gif" width=350>


---
# About the project

This project is divided in four main files:

**1.** [**GetImages.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/GetImages.py):
    
This program is responsible to gather the images that will be used to train and test the **Neural Network**. Each time the user presses 's', a new batch of images will be saved with the label according to the text displayed on top of the video feed. Only the frame inside the white rectangle will be registered. Two GIF images showing the UI, as well as the saved files can be found below.

**2.** [**NetworkTraining.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/NeuralNetworkTraining.py):
    
This program will organise the images collected by the previous program and train a **Convolutional Neural Network** to recognise the three classes mentioned in the previous section. The evolution of the training after each **epoch** can be seen in the terminal window and, at the end, a **Confusion Matrix** will represent the accuracy of the predictions made by the **trained Neural Network** when taking the images in the **images/testImages** directory as input. A better explanation about the evolution of the training and testing of the [**HandGestureRecognition_UpDownNone_20k.h5**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5) **Neural Network** can be found at the end of this section. 

**3.** [**HandGestureRecognition_Test.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition_Test.py):
    
This program can be used to test a **trained Neural Network**, using the images in the **images/testImages** directory as input and visualise its **Confusion Matrix** without the need of running **NetworkTraining.py** (NetworkTraining.py can take a long time to finish running).

**4.** [**HandGestureRecognition.py**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/HandGestureRecognition.py):
    
This is the program used to recognise the three classes previously mentioned. A brief explanation, together with two GIF images can be found in the first section of this file.


<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/GetImages.gif" width=350> <img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/imageFiles.gif" width=350>


For the particular case of the [**HandGestureRecognition_UpDownNone_20k.h5**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5) **Neural Network**, a set of **18.000** images were used to train the network, **3.000** images were used as validation to track the progress of each epoch (**val_loss** and **val_accuracy**) and **3.000** images were used to test the result.

The training progress, as well as the **Confusion Matrix** can be seen below. For the specific set of images used to test the **Neural Network**, it was capable of predicting **1.000 out of 1.000 thumbsUp** images, **998 out of 1.000 thumbsDown** images and **987 out of 1.000 noHand** images (as seen in the confusion matrix).

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/NeuralNetwork_TrainingProgress.PNG"> 

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/ConfusionMatrix.png">


---
# How to run the project

To run the project you will need **Python 3 (x86-64 bits)** (the project was developed using **Python 3.8.5**). To check your Python version you can open the **Windows Command Prompt** and run command below:
> python --version

After having Python installed:

**1.** Copy the Python files (**GetImages.py**, **NetworkTraining.py**, **HandGestureRecognition_Test.py**, **HandGestureRecognition.py** and **RunProject_CreateVENV.py**), together with **requirements.txt** and the **TrainedNeuralNetworks** folder to a directory in your computer (everything should be in the the same directory).

**2.** Run (double click) **RunProject_CreateVENV.py**. This file will create a virtual environment, install all the requirements listed in **requirements.txt** and then run **HandGestureRecognition.py** automatically.


**Alternative option to run the project:**

After following **step 1** of the list above, open the **Windows Command Prompt** and run the following commands:

**1.** Create and activate a virtual environment (optional):
> python -m venv HandGestureRecognition_venv

> HandGestureRecognition_venv\Scripts\activate.bat

**2.** Install the requirements listed in **requirements.txt** (if you choose to skip the previous step, the requirements will be installed locally in your computer):
> pip install -r requirements.txt

**3.** Run **HandGestureRecognition.py**:
> HandGestureRecognition.py


**ATENTION!** - Depending on your computer and camera settings, it might be necessary to change **line 22** of the code in **HandGestureRecognition.py** (image below).

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/CameraCode.PNG">



## Testing the Neural Network

In case you want to test the [**HandGestureRecognition_UpDownNone_20k.h5**](https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/TrainedNeuralNetworks/HandGestureRecognition_UpDownNone_20k.h5) **Neural Network**, you can copy the [**images/**](https://github.com/patrickmetzner/AI_HandGestureRecognition/tree/master/HandGestureRecognition/images/testImages) folder to the **SAME DIRECTORY** where all the **.py** files are located and run **HandGestureRecognition_Test.py**.

Inside the [**images/**](https://github.com/patrickmetzner/AI_HandGestureRecognition/tree/master/HandGestureRecognition/images/testImages) folder you can find 30 test images (10 for each image class). 

In relationship with the **.py** files, the images should be in the following paths:

**..\images\testImages\noHand**

**..\images\testImages\thumbsDown**

**..\images\testImages\thumbsUp**


To run **HandGestureRecognition_Test.py** you can open the **Windows Command Prompt** and run the following commands:

**1.** Activate a virtual environment created previously (if opted to create virtual environment):
> HandGestureRecognition_venv\Scripts\activate.bat

**2.** Run **HandGestureRecognition_Test.py**:
> HandGestureRecognition_Test.py

Observation: If you opted to install the requirements locally in your computer, you can just double click **HandGestureRecognition_Test.py**)


## Training your own **Neural Network**

In case you want to gather your own images and train your own **Neural Network**, you can run **GetImages.py** until you get a reasonable number of images (at least 1.000 of each class is recommended), open the **NetworkTraining.py** file and change the variables in the **lines 18**, **19** and **20** accordingly (image below). 

**Note that the total number of images gathered must be greater that the sum of these three variables.**

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/TrainingVariables.PNG">


The new Neural Network will be saved as **HandGestureRecognition_New.h5** inside the **TrainedNeuralNetworks** directory.

To run **HandGestureRecognition.py** with your new Neural Network, change the name of the loaded Neural Network in **line 26** accordingly (image below).

<img src="https://github.com/patrickmetzner/AI_HandGestureRecognition/blob/master/HandGestureRecognition/README_images/LoadNeuralNetwork.PNG">


**ATTENTION!** - Depending on your computer and camera settings, it might be necessary to change **line 22** of the code in **HandGestureRecognition.py** and **line 28** of the code in **GetImages.py** (image shown previously).


# Docker container (work in progress)

A Docker container was generated with the files and settings needed to run this project. It can be accessed by the following commands:

**1.** Pull the container image from DockerHub: 
> docker pull patrickmetzner/hand-gesture-recognition-container 

**2.** Use the pulled image to run a docker container in interactive mode:
> docker run -it -e DISPLAY=**[HOST IP ADDRESS]**:0 --name hand-gesture-recognition-container hand-gesture-recognition-image

Note that you need to enter the digits of your computer's IP address in place of **[HOST IP ADDRESS]** in order for the container to access the computer's display. On Windows, the IP address can be quickly found with the following command:
> ipconfig

Look for the **IPv4 Address**. It should be something like: XXX.XXX.X.XX 

**3.** The container's bash command is ready to be used. 

To test the **Neural Network**, you can run the following command in the container's CLI:
> python HandGestureRecognition_Test.py

----

**Observations:**
**1.** When running this container on Windows, you need the **XLaunch** app in order for the container to access the computer's display.

The app is part of the **VcXsrv Windows X Server** and can be downloaded [here](https://sourceforge.net/projects/vcxsrv/).

**When running XLaunch, make sure to check the "Disable access control" box at the "Extra settings" window.**

**2.** Just the **HandGestureRecognition_Test.py** script will work on Windows. The other scripts need to access the computer's WebCam and the container is not set up to that (yet).

on Linux, it might be possible to overcome this problem with the following command: (note that I have not tested this command yet)
> docker run -it -e DISPLAY=[HOST IP ADDRESS]:0 **--device=/dev/video0:/dev/video0** --name hand-gesture-recognition-container hand-gesture-recognition-image
