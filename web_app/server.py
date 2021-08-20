import tempfile
import time

import cv2
import streamlit as st
from src.constants import CHECKPOINTS_PATH, OUTPUT_FILE
from src.models.basic_cnn.basic_cnn_class import BasicCNN
from src.models.mediapipe.mediapipe_class import Mediapipe
from src.utils import inference_transformations

from web_app.utils import setup_annotation, setup_parameters

DEMO_VIDEO = "/workdir/web_app/demo.mp4"

model_name, detection_confidence = setup_parameters()
model_CNN = BasicCNN(CHECKPOINTS_PATH, inference_transformations, OUTPUT_FILE, detection_confidence)
model_mediapipe = Mediapipe(OUTPUT_FILE, detection_confidence)
if model_name == "BasicCNN":
    model = model_CNN
else:
    model = model_mediapipe

stframe = st.empty()
video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])
tfflie = tempfile.NamedTemporaryFile(delete=False)

if not video_file_buffer:
    vid = cv2.VideoCapture(DEMO_VIDEO)
    tfflie.name = DEMO_VIDEO
else:
    tfflie.write(video_file_buffer.read())
    vid = cv2.VideoCapture(tfflie.name)

fps, width, height, fps_input, kpi1_text, kpi2_text, kpi3_text = setup_annotation(vid, tfflie)

prevTime = 0
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = True

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
    kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{height}</h1>", unsafe_allow_html=True)
    kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    frame = model.predict_over_image(frame, return_image=True, save_image=False)

    stframe.image(frame, channels="RGB", use_column_width=True)

vid.release()
