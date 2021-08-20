facial_landmark
==============================

Real time facial landmark on CPU

Please download this dataset : https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points and unzip the file in data/raw to end up with :

Face detection part by : https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/tflite

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── api                <- Api container 
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │       ├── face_images.npz
    │       ├── face_input.jpg
    │       ├── facial_keypoints.csv     
    │
    ├── modeling           <- Modeling container 
    │
    ├── tensorboard        <- Tensorboard container
    │
    ├── tensorboard_logs   <- Logs of training, contains demo model
    │
    ├── docker-compose.yml <- to run all containers
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------