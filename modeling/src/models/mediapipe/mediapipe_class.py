import cv2
import mediapipe as mp
from src.models.base_model import BaseModel

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class Mediapipe(BaseModel):
    def __init__(self, output_file, min_detection_confidence: float = 0.5):
        super().__init__(output_file)
        self.model = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=min_detection_confidence
        )

    def predict(self, x):
        # should return coordinates
        raise NotImplementedError("Not implemented yet.")

    def predict_over_image(self, x, return_image: bool = False, save_image: bool = True):
        results = self.model.process(x)
        if results.multi_face_landmarks:
            annotated_image = x.copy()
            for face_landmarks in results.multi_face_landmarks:
                # print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

            if save_image:
                cv2.imwrite(self.output_file, annotated_image)

            if return_image:
                return annotated_image
