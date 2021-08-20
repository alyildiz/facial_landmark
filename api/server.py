import cv2
import jsonpickle
import numpy as np
from flask import Flask, request
from PIL import Image
from src.constants import CHECKPOINTS_PATH, OUTPUT_FILE
from src.models.basic_cnn.basic_cnn_class import BasicCNN
from src.models.mediapipe.mediapipe_class import Mediapipe
from src.utils import inference_transformations

app = Flask(__name__)

model_CNN = BasicCNN(CHECKPOINTS_PATH, inference_transformations, OUTPUT_FILE)
model_mediapipe = Mediapipe(OUTPUT_FILE)


@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.args.get("model_name")
    data = request.files["file"]
    img = Image.open(data)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

    if model_name == "BasicCNN":
        model = model_CNN
    elif model_name == "Mediapipe":
        model = model_mediapipe
    else:
        raise ValueError("Model type not supported.")

    image = model.predict_over_image(img_cv2, return_image=True)
    response = jsonpickle.encode(image)

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
