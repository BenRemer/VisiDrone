# Installing the VisiDrone project
This document covers how to install the VisiDrone project, its dependencies, and setting up the environment.

- For general information about the VisiDrone project, please see [`README.md`](README.md).
- For training new models, exporting models, and adding them to the project visit [`OBJECT-DETECTION.md`](OBJECT-DETECTION.md).

## Prerequisites
[Python 3.5+](https://www.python.org/downloads/)
- imageio
- jupyter
- lxml
- matplotlib
- numpy
- onvif-zeep
- Pillow

[Tensorflow](https://www.tensorflow.org/install/)
- Tensorflow is our library of choice for machine learning and object detection. 
- The [CPU version](https://www.tensorflow.org/install/) can be done via the command: ```pip install tensorflow``` or ```py -3 -m pip install tensorflow```. Make sure your Python version is 3.5 to 3.7 because those are the supported versions of Python (as of April 2019). 
- The [GPU version](https://www.tensorflow.org/install/gpu) can also be installed if your computer has an [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) with and [NVIDIA Compute Capability of over 3.5](https://developer.nvidia.com/cuda-gpus). As described by the [link](https://www.tensorflow.org/install/gpu) the software requirements are [updating your NVIDIA GPU drivers](https://www.nvidia.com/drivers), [installing CUDA Toolkit](https://developer.nvidia.com/cuda-zone), and the [cuDNN SDK](https://developer.nvidia.com/cudnn). The all these instructions (and Linux/Docker instructions) are in the [official Tensorflow install guide](https://www.tensorflow.org/install/gpu). To complete the setup of tensorflow-gpu for Windows make sure too look at the [very bottom section of the official install guide](https://www.tensorflow.org/install/gpu). For any errors, consult the troubleshooting section and google the error messages. 
- A good way to test that your Tensorflow installation has worked is by going into the Python interpreter ```python``` or ```py -3```. Then type in ```import tensorflow as tf``` and press enter. If no error occurs the installation has been properly installed. 

[Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- Required for performing object detection with Tensorflow.
- A good guide for Windows users is this [Medium article](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b). 
It also describes how to install the COCO API and Protocol Buffers. A few notes with the [Medium article](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b) are when not installing with Conda (which we did not use), ignore any instructions that describe doing anything with Conda (don't add any values with anaconda to PYTHONPATH). 
- Linux instructions are covered in the [official installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). Using a Cloud VM that is pre-set and compiled with the common machine learning frameworks and APIs is also an option (in fact its probably better than our local machine setup).
- For Mac this [Medium article](https://medium.com/@viviennediegoencarnacion/how-to-setup-tensorflow-object-detection-on-mac-a0b72fbf470a) is helpful.
- The common way to test if the Tensorflow Object Detection API is properlly installed is by running the Jupyter Notebook found within `models/research/object_detection`. It is called `object_detection_tutorial.ipynb`. To run this simply navigate to the directory and run jupyter notebook and once Jupyter Notebook begins running on localhost click and open it. Next simply run each step until an image appears at the end. Be sure to change the line ```from matplotlib import pyplot as plt``` to ```import matplotlib.pyplot as plt```. Also make sure to wait until the current cell finishing running (an number appears instead of an asterisk) before running the next cell.  

[COCO API](https://github.com/cocodataset/cocoapi)
- Required for working some functions of the Tensorflow Object Detection API (namely eval.py).
- There are a couple ways to install the COCO API for use with the Tensorflow Object Detection API. 
- The way to typically install it is described in a [Medium article describing how to Intall the Tensorflow Object Detection API on Windows](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b) as Step 6. 
- For Windows on Python 3+, my preferred way is to install it through a [fork of cocodataset/cocoapi](https://github.com/philferriere/cocoapi).
The prerequisite is to [install Visual C++ build tools as described](https://github.com/philferriere/cocoapi).
Afterwards you run the command: ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI ```.

[Protocol Buffers v3.4.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)
- Required for setting up the Tensorflow Object Detection API.

## Troubleshooting
- Tensorflow-GPU installation: Sometimes if you install the wrong/unsupported versions of the CUDA Toolkit and the cuDNN SDK you can run into issues. Make sure your software installations are compatible. [This StackOverFlow post might be helpful](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible).
- Tensorflow Object Detection API in General: A problem that sometimes appears is where all the object-detection libraries for Tensorflow aren't recognized. This occurs typically occurs because environment variables haven't been properly added for where the Tensorflow Object Detection API has been cloned to. 

  
