import numpy as np
import pandas as pd
from src.constants import FACE_IMAGES_PATH, FACIAL_KEYPOINTS_PATH


def load_data():
    df_facial_keypoints = pd.read_csv(FACIAL_KEYPOINTS_PATH)

    # we only keep the rows where we have 30 data points
    df_facial_keypoints = df_facial_keypoints.dropna()
    idx_to_keep = df_facial_keypoints.index.tolist()
    df_facial_keypoints = df_facial_keypoints.reset_index(drop=True)

    db_face_images = np.load(FACE_IMAGES_PATH)["face_images"]
    db_face_images = db_face_images[:, :, idx_to_keep]

    db_face_images = np.moveaxis(db_face_images, -1, 0)
    db_face_images = np.asarray(db_face_images).reshape(db_face_images.shape[0], 96, 96)

    return df_facial_keypoints, db_face_images
