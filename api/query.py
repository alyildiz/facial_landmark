import jsonpickle
import requests
from src.constants import DEMO_IMAGE_FILE

files = {
    "file": ("image", open(DEMO_IMAGE_FILE, "rb")),
}

params = {"model_name": "Mediapipe"}

url = "http://0.0.0.0:5000/predict"

r = requests.post(url, params=params, files=files)
result = jsonpickle.decode(r.text)
print(result.shape)
