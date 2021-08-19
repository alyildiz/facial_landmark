import argparse

from src.load_data import load_data
from src.pl_datamodule import DataModule
from src.pl_model import BasicCNNModel
from src.train_model import train_model
from src.utils import basic_transformations, preprocess, train_test_split
from torchsummary import summary


def main(epochs):
    df_facial_keypoints, db_face_images = load_data()
    db_face_images, df_facial_keypoints = preprocess(db_face_images, df_facial_keypoints)
    train_x, train_y, test_x, test_y = train_test_split(db_face_images, df_facial_keypoints, split_size=0.2)
    train_x, train_y, val_x, val_y = train_test_split(train_x, train_y, split_size=0.2)

    module = DataModule(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        transform=basic_transformations,
        batch_size=32,
    )

    model = BasicCNNModel(input_shape=(1, 96, 96))
    # print the model summary
    summary(model, input_size=(1, 96, 96), device="cpu")
    train_model(model, module, max_epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default=100, type=int, help="the base")
    args = parser.parse_args()
    main(epochs=args.epochs)
