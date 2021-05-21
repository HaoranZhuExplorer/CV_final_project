# CV_final_project

## Clone the repository

```
git clone https://github.com/HaoranZhuExplorer/CV_final_project.git
```

# Lane Detection

We change the code based on https://github.com/ZJULearning/resa, which is an official codebase to train a model with TuSimple dataset
And we insert ERFNet architecture into this codebase. For data downloads and set up the environment, please see https://github.com/ZJULearning/resa.

## Training

For training, run

```Shell
python lane_detection/main.py [configs/path_to_your_config] --gpus [gpu_ids]
```

# License Plate Recognition

See https://github.com/HaoranZhuExplorer/CV_final_project/tree/main/license_plate_recognition
