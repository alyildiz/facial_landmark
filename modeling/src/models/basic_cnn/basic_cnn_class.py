import cv2
from src.constants import FACE_DETECTION_MODEL_PATH
from src.models.base_model import BaseModel
from src.models.basic_cnn.pl_model import BasicCNNModel
from src.models.face_detection.TFLiteFaceDetector import UltraLightFaceDetecion


class BasicCNN(BaseModel):
    def __init__(self, checkpoints, transformations, output_file):
        super().__init__(output_file)
        self.model = BasicCNNModel.load_from_checkpoint(checkpoints)
        self.transformations = transformations

    def predict(self, x):
        x = x.unsqueeze_(0)
        return self.model(x)

    def predict_over_image(self, x, return_image: bool = False):
        n, m = x.shape[:2]
        fd = UltraLightFaceDetecion(FACE_DETECTION_MODEL_PATH, conf_threshold=0.6)
        boxes, scores = fd.inference(x)
        for result in boxes.astype(int):
            top_left = (result[0], result[1])
            bottom_right = (result[2], result[3])
            height = bottom_right[1] - top_left[1]
            width = bottom_right[0] - top_left[0]

            face_extract = x[top_left[1]: top_left[1] + height, top_left[0]: top_left[0] + width]
            face_extract = self.transformations(face_extract)
            preds = self.model(face_extract.unsqueeze_(0)).detach().numpy()[0]

            for i in range(0, 30, 2):
                x_data = preds[i]
                y_data = preds[i + 1]
                x = cv2.circle(
                    x, (int(result[0] + x_data * width), int(result[1] + y_data * height)), 2, (0, 255, 0), 2
                )

        cv2.imwrite(self.output_file, x)

        if return_image:
            return x
