import argparse

import cv2
from src.models.basic_cnn.basic_cnn_class import BasicCNN
from src.models.mediapipe.mediapipe_class import Mediapipe
from src.utils import inference_transformations

from src.constants import DEMO_IMAGE_FILE, CHECKPOINTS_PATH, OUTPUT_FILE

def main(model_name, image_file, checkpoints, output_file):
    im = cv2.imread(image_file)

    if model_name == "BasicCNN":
        model = BasicCNN(checkpoints, inference_transformations, output_file)
    elif model_name == "Mediapipe":
        model = Mediapipe(output_file)
    else:
        raise ValueError("Model type not supported.")

    model.predict_over_image(im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="BasicCNN",
        type=str,
        help="""Use BasicCNN for trained model or Mediapipe for SOTA performances""",
    )
    parser.add_argument(
        "-i", "--image_file", default=DEMO_IMAGE_FILE, type=str, help="""Path to image file"""
    )
    parser.add_argument(
        "-c",
        "--checkpoints",
        default=CHECKPOINTS_PATH,
        type=str,
        help="""Path to saved model""",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        default=OUTPUT_FILE,
        type=str,
        help="""Path to saved image""",
    )
    args = parser.parse_args()
    main(model_name=args.model_name, image_file=args.image_file, checkpoints=args.checkpoints, output_file=args.output_file)
