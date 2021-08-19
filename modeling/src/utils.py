import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torchvision import transforms


def plot_face(
    db_face_images: np.ndarray,
    df_facial_keypoints: pd.DataFrame,
    idx: int,
    rescale_pixels: bool = False,
    rescale_position: bool = True,
):
    frame = db_face_images[idx].copy()
    if rescale_pixels:
        frame /= 255

    for i in range(0, df_facial_keypoints.shape[1], 2):
        x_coordinates = df_facial_keypoints.iloc[idx, i]
        y_coordinates = df_facial_keypoints.iloc[idx, i + 1]
        if rescale_position:
            x_coordinates *= 96
            y_coordinates *= 96
        frame = cv2.circle(frame, (int(x_coordinates), int(y_coordinates)), 1, (0, 255, 0), 1)

    cv2.imshow("img", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train_test_split(db_face_images: np.ndarray, df_facial_keypoints: pd.DataFrame, split_size: float):
    df_facial_keypoints = shuffle(df_facial_keypoints)
    n = int(df_facial_keypoints.shape[0] * (1 - split_size))
    train_y = df_facial_keypoints.iloc[:n, :]
    idx_to_keep = train_y.index.tolist()
    train_y = train_y.reset_index(drop=True)

    train_x = db_face_images[idx_to_keep]

    split_y = df_facial_keypoints.iloc[n:, :]
    idx_to_keep = split_y.index.tolist()
    split_y = split_y.reset_index(drop=True)

    split_x = db_face_images[idx_to_keep]

    return train_x, train_y, split_x, split_y


def preprocess(x, y):
    # rescale pixels color from 0 to 1
    x_copy = x.copy()
    x_copy /= 255

    # rescale position from 0 to 1 because image size is 96x96
    y_copy = y.copy()
    y_copy /= 96

    return x_copy, y_copy


basic_transformations = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
