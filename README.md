# VisiDrone
Junior Design Project

This repo is a Georgia Tech Junior Design Capstone Project for the 2018-2019 school year.

## Release Notes
### Frontend Repository
The repository that contains our frontend code is [here](https://github.gatech.edu/jliu614/visidrone). The usage for the frontend has not changed since we were given the original code. We have added a drone_icon_layer to the map. The frontend can now read a GeoJSON object and display a notification on the map corresponding to a drone.
### New software features for this release
- Detect drones from the following sources:
  - object-detection/test_images folder (file size must fit the model specifications)
  - a single snapshot from a connected Avigilon H4 Multisensor
  - a continuous stream of images from a connected Avigilon H4 Multisensor

### Known bugs and defects
- The current model is not perfect at dectecting drones and is subject to change.
## Install Guide
Install VisiDrone by first cloning the GitHub repository: ```git clone https://github.gatech.edu/bremer3/VisiDrone.git```.

- Find the installation instructions for VisiDrone in [`INSTALL.md`](INSTALL.md).
- Find object-detection related instructions in [`OBJECT-DETECT.md`](OBJECT-DETECT.md).

### Usage
To process an image from a connected Avigilon H4 Multisensor:
```python3 Visidrone.py```

To process a continuous stream of images from a connected Avigilon H4 Multisensor:
```python3 Visidrone.py stream```

To process all images in the object-detection/test_images folder:
```python3 Visidrone.py test```

