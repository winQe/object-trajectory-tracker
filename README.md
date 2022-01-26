# MYNT-EYE Object Trajectory Tracker

## Overview

An object trajectory tracker developed for the MYNT-EYE D RGB-D Camera. Performs object detection on selected object class, and draws bounding box & trajectory path. Also outputs the object coordinates and distance to the object. By default, it uses **YOLOv4** pre-trained weight and only detects humans. But it can easily be customized. See the **Usage** section below.

## Installation

### Hardware Requirements

* MYNT EYE D-1000 RGB-D Camera

### Dependencies

* MYNT-EYE-D SDK
* OpenCV 4.1 with DNN module

### Building
In order to install, clone the repository and compile the package.

    git clone https://github.com/winQe/object-trajectory-tracker
    mkdir build & cd build
    cmake ..
    make

### Download weights
The yolov3-tiny.weights are included in this repository, however the pre-trained yolov4.weights must be downloaded manually.
    
    cd ../darknet/
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


## Usage
Before using, there are a few things to take note of 
1. [main.cpp](main.cpp) line 24-26 : Make sure the path to the yolo weights, config, and coco.names files are correct. Use absolute path to avoid any errors.
2. [CDNeuralNet.cpp](CDNeuralNet.cpp) line 75 & 104 : You can change the minimum detected confidence and the detected classes here. Class Id depends on the line number of coco.name file. Currently only detects object of classid 0.

To run the code
    
    cd build
    ./object_tracker

## Limitations
* In the current state, can only track one single object at a time. If there is more than one object, the trajectory path will become very messy.
* Detects object in real-time but there might be a few miliseconds delay when drawing the bounding boxes.
* Only performs object classification, so it is unable to differentiate between objects with the same class Id.

## Future works
* Implement other algorithm such as DEEP-SORT to perform the tracking.