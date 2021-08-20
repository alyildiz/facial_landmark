facial_landmark
==============================

Real time facial landmark on CPU

If you want to retrain the model download this dataset : https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points and unzip the file in data/raw to end up with :

Face detection part by : https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/tflite

API
==============================
To run the API : ```docker-compose up api```
The flask API will run on the port 5000, to query the API, follow the example in ```api/query.py``` 
The ```/predict``` endpoint will return you the input image annotated using either the BasicCNN or the Mediapipe model.

Tensorboard
==============================
To run Tensorboard : ```docker-compose up tensorboard```
Go to ```localhost:6006``` to visualize the interface.

Modeling
==============================
To run the container : ```docker-compose up modeling```
Use ```docker exec -it modeling bash``` to get inside the container. From there, there are 2 endpoints :
- ```/workdir/bin/inference.py``` to use inference over an image using one model
- ```/workdir/bin/train_and_save.py``` to train the BasicCNN model (downloading the dataset is mandatory)


 
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
    ├── web_app            <- Web app container
    │
    ├── docker-compose.yml <- to run all containers
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
