# Anthropometric Measurement Estimation using OpenPose Algorithm

![teaser figure](images/teaser.png)
**Anthropometric measurement estimation algorithm:** From left to right--original image of patient; undistorted image; keypoint estimation using OpenPose; anthropometric measurements (projections). </p> 

<br/>

This repository contains the code for an anthropometric measurement estimation algorithm using the OpenPose model developed by CMU Perceptual Computing Lab. The algorithm estimates the neck, torax, abdomen and waist measurement projections of male patients using single frontal images. Future work will focus on developing regression models to estimate the circunference of the selected areas based on the frontal projections. The circunferences, and even the projections, can be used by medical doctors in occupational health clinics to evaluate the patient's capability on performing certain jobs.

## Description

### 1. Hardware

The anthropometric measurement system consists of a camera that acquires images with a resolution of 1280x720 pixels, a Raspberry Pi 4, used to acquire the images, and a remote server that processes the images and estimates the measurements. The code in this repository runs on the remote server.

![hardware](images/hardware.png)

### 2. Image acquisition protocol

An image acquisition protocol is used to standardize all images. The camera is located at a distance of 1.27 meters from the ground and perpendicularly at a distance of 1 meter from the patient. The patient must remove his shirt and extend his arms at an angle of 45 degrees with respect to the torso. The acquisition system then acquires a frontal image of the patient.

![acquisition](images/acquisition.png)

### 3. Algorithm

The algorithm estimates five anthropometric measurements: (1) neck, (2) torax, (3) abdomen, (4) waist and (5) hips. In order to accomplish this, the algorithm uses six stages:

![algorithm](images/algorithm.png)

#### 3.1. Image segmentation

The patient's body is segmented using color thresholding in the HSV color space. Then, the mahalanobis distance is used to isolate the patient's color skin, eliminating the background and creating a binary mask. The mask is then refined by applying morphology operations.

![segmentation](images/segmentation.png)

#### 3.2. Keypoint identification

The algorithm identifies the patients, keypoints using MCU Perceptual Computing Lab's OpenPose model. This algorithm uses convolutional neural networks (CNNs) to identify and localize several parts of the body in color images. The OpenPose model first generates feature maps using a VGG-19 network. Then, the feature maps are used as input to a second CNN, which generates part affinity maps. Lastly, a third CNN is used to produce probability maps for each body part.

#### 3.3. Keypoint validation

The OpenPose model is trained on two datasets: MPII and COCO. In this project we use both in order to improve the accuracy of the algorithm. For example, if one of the model trained with one dataset is not able to identify one of the body parts, the other one can be used as a substitute.

![models](images/coco_vs_mpii.png)

#### 3.4. Anthropometric projections