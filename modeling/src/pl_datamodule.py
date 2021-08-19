import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class FaceData(Dataset):
    def __init__(self, db_face_images, df_facial_keypoints, transform):
        self.db_face_images = db_face_images
        self.df_facial_keypoints = df_facial_keypoints
        self.transform = transform

    def __len__(self):
        return self.df_facial_keypoints.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.db_face_images[idx]
        label = self.df_facial_keypoints.iloc[idx, :].values

        if self.transform:
            image = self.transform(image)
        label = torch.FloatTensor(label)

        return image, label


class DataModule(pl.LightningDataModule):
    def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y, transform, batch_size):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.transform = transform
        self.batch_size = batch_size

    def prepare_data(self):
        self.train = FaceData(db_face_images=self.train_x, df_facial_keypoints=self.train_y, transform=self.transform)
        self.valid = FaceData(db_face_images=self.val_x, df_facial_keypoints=self.val_y, transform=self.transform)
        self.test = FaceData(db_face_images=self.test_x, df_facial_keypoints=self.test_y, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
