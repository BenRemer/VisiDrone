# Object Detection Instructions
This document covers how to train new models, exporting trained models for use, and modifying the model within our object detection pipeline.
- For general information about the VisiDrone project, please see [`README.md`](README.md).
- For installation information go to [`OBJECT-DETECTION.md`](INSTALL.md).

# Acquiring More Data
- Collecting data for training an object detection model is an essential part of the process. 
- An important first consideration is the resolution you want for your dataset. Keep in mind that larger images require more GPU usage and training time. The downside of smaller images is that drones are harder to capture at a distance. We chose the resolution (600x338).
- The next step is to obtain a dataset of images with your chosen resolution. 
- The way we handled obtaining a dataset of drones was by using Flickr and searching for "DJI Phantom". We simply downloaded the the images and resized images close to the desired size while keeping the aspect ratio (to prevent warped images) into a standardized size (600x338). If the images still aren't the same size, crop the images down to the correct size. This process was time intensive and laborious and netted us 169 images of drones (119 for training, 50 for evaluation).
- A better way would be to ask a drone pilot to fly a drone and take pictures with the camera (Avigilon H4 Multisensor). Extracting images from the camera is easy and all of the image sizes are the same. Just make sure the images you choose are from diverse angles and lighting conditions, so that the dataset contains a variety of examples. Having too many similar images will result in a less generalizable model.      
- After collecting the dataset split the dataset into a training set and a validation set. We chose a 70-30 split (training-validation). The exact proportion of the training and validation sets is up to your discretion. 

## Labeling Images
- Before training can begin, the images within the dataset must be labeled. 
- This can be done with a variety of tools, but we chose the [LabelImg tool](https://github.com/tzutalin/labelImg). The installation guide is in the [GitHub repository](https://github.com/tzutalin/labelImg). 
- Using the LabelImg tool open the folder with the dataset images. Then choose an output folder for where the xml files will be saved to. Labeling images is as simple as putting "drone" as the default label, drawing the bounding box where the drone is, and saving. The labeled image saves as an xml file in the PASCAL 2012 VOC format in the output folder.

## Convert to TF-Record
- After labeling both the training set and the validation set, it is necessary to convert both into a format called TF-Record. This format is needed by the Tensorflow to train a given object detection model. 
- Using the `xml_to_csv.py` script after modifying the `image_path` variable to where your xml files are stored creates a `drone_labels.csv` file which can be renamed to whatever (ex. `train_labels.csv` or `val_labels.csv`).
- After a csv file is generated, the `generate_tfrecord.py` script is used to convert the csv to a TF-Record. There's already a usage guide within the initial comments of `generate_tfrecord.py`. But essentially you run the command `python generate_tfrecord.py --csv_input="path-to/train_labels.csv" --output_path="where-you-want-save/train.record" --image_dir="path-to-training-images"` from the models folder (where you installed the Tensorflow Object Detection API) and the TF-Record file is created.
- You should be at the point where you have both a `train.record` and a `val.record`.  

## Training New Models
Now the actual training can begin. The instructions for training were adapted from an [article on "Creating Your Own Object Detector"](https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85).
1. Move the script `train.py` from the legacy folder (`models/research/object_detection/legacy`) and move it to the object_detection folder (`models/research/object_detection`).
2. Move the script `eval.py` from the legacy folder (`models/research/object_detection/legacy`) and move it to the object_detection folder (`models/research/object_detection`).
3. Create a train_dir (for where you want to send the training metadata) folder and a eval_dir (for where you want to send the evaluation metadata) folder.
4. [Understand configuration files](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). You can choose different object detection models and change many hyperparameters for training in the configuration file. Our model's config file is found in `VisiDrone/object-detection/config` as `pipeline.config`. Another important feature of config files is that you can specify a `fine_tune_checkpoint`. Checkpoints are essentially the metadata (weights) that corresponds to the saved state of a trained model. This can be used for transfer learning, which makes training much faster than going from scratch. Our model uses the [faster_rcnn_inception_v2_coco_2018_01_28 model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) as a checkpoint.
5. Write a new configuration file from scratch or use a previously used configuration file as a template (our configuration file is found in `VisiDrone/object-detection/config` as `pipeline.config`). Make sure to edit the train_input_reader (input_path = where train.record is located) and eval_input_reader (input_path = where eval.record is located) sections. Change the label_map_path to the path of where your label_map.pbtxt is located locally. Parameters like `num_examples` in `eval_config` should also be changed to their appropriate values.   
6. Run `train.py` and `eval.py`. In one command line, run: 
    ``` 
        python train.py \
        --logtostderr \
        --train_dir="path/to/train_dir" \
        --pipeline_config_path="path/to/config_file.config" 
    ```
    In another command line run:
    ```
        python eval.py
        --logtostderr \
        --checkpoint_dir="path/to/train_dir" \
        --eval_dir="path/to/eval_dir" \
        --pipeline_config_path="path/to/config_file.config"
    ```
    If there are any problems running both scripts together [consult this GitHub issue](https://github.com/tensorflow/models/issues/1854#issuecomment-329410804) or more specifically [this fix](https://github.com/tensorflow/models/issues/1854#issuecomment-329410804).
7. Run `tensorboard --logdir="path/to/eval_dir"` in one command line and `tensorboard --logdir="path/to/train_dir"` in another command line. This allows for you to view the status of the model being trained based off of different metrics at `localhost:6006`. If an error occurs when running tensorboard about an `Invalid format string`, you the [fix in this StackOverFlow question](https://stackoverflow.com/questions/54814113/invalid-format-string-tensorboard). 
8. Wait until both total loss and ``mAP@0.5IOU`` reach satisfying values and then stop training the model. The optimal training time varies a lot depending on model, hyperparameters, and hardware.   

## Exporting Models for Use
After training has been completed, all of the metadata (checkpoints) are located in your chosen train_dir folder. 
1. Run the script `export_inference_graph.py` found in `models/research/object_detection`. It is run like:
    ```
    python export_inference_graph.py \
    --pipeline_config_path="path/to/config_file.config"
    --trained_checkpoint_prefix="path/to/train_dir/model.ckpt-XXXX" 
    --output_directory="path/to/where-you-want-the-model-to-save-to"
    ```
    Note that for `model.ckpt-XXXX` the XXXX is replaced with the highest value found for any ckpt files in your train_dir. For Windows users, don't use "\\" when writing out the path. Using a single "\" should be fine.  
2. A `frozen_inference_graph.pb` should appear where you directed the `output_directory` to be. Simply replace the current `frozen_inference_graph.pb` and the trained model can now be run.
This is the general procedure for re-training new models and testing them out.


## Advice for the Future
1. Use a Cloud VM (with pre-set and compiled machine learning libraries) for Both Training and Production Deployment. 
    - The best way to make this project maintainable in the future. 
    - The only downside is that it costs money (try to find estimates on the cost, use free trial credits, and ask GTPD for possible reimbursements from their budget). 
    - Cloud VMs are also useful because you can choose ones with both Linux and an NVIDIA GPU, which simplifies the time spent on setup and debugging significantly. 
    - VMs or Docker containers that run on a laptop or desktop cannot access the host computers GPU.
2. Learn How to Use and Deploy Docker Containers. 
    - Using Linux environments will simplify work. 
3. Use the Tensorflow Serving Library
    - For putting the object detection model into production, the Tensorflow Library TF-Serving is the best way to do so. 
    - We ran into both errors setting the TF-Serving environment up and a time constraint, but it is still the best way to put any machine learning model into production. 
    - My problems with setting up the TF-Serving environment was possibly due to a lack of expertise with working with Docker. I had a TF-Serving Docker container with our object detection model loaded on it and a client script from my host computer that attempted to send the model a object detection request. For some reason there was a "Null Socket Error". 
4. Hyperparameter Tuning and Better Datasets 
    - Optimizing a variety of factors could improve both training performance and accuracy. 
    - Obtaining a larger dataset of images could also help (we had a dataset of 169 images with 119 for training and 50 for validation, which were all taken from Flickr). 
    - Asking to take pictures of drones (from diverse angles and lighting conditions) would help a lot. 
    - The downsides are that the labelling of images and the training of the model would be more time consuming. 
    - If one has the appropriate computing power, larger images could also be used in training. We used images scaled down to (600x338) because of GPU resource constraints and becausse it was faster for training.    
5. Inference Speed Improvements 
    - Can be sped up by using [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) with a setup tutorial [here](https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa).
    - Our Faster R-CNN object detection model is on the slower end of algorithms. Possibly evaluating different models could speed up the inferencing step (probably focus on accuracy over speed). 
6. Notes
    - A bunch of notes for working with Tensorflow Object detection are located in the Notes folder. There are plenty of links to tutorials or StackOverFlow posts that might be useful.  

## Troubleshooting
- `ImportError: No module named 'tensorflow.python.saved_model.model_utils'` after trying to run `train.py`. This error was fixed by [consulting GitHub issues](https://github.com/tensorflow/tensorflow/issues/27079) and running `pip uninstall tensorflow_estimator` and `pip install tensorflow_estimator`.
- With any `google.protobuf.text_format.ParseError` look to make sure your directories are spelled correctly as both command line arguments. Also make sure to use "\\" for "\" when writing out paths for Windows computers (they are often parsed as escape characters). 
