import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class BasicCNNModel(pl.LightningModule):
    def __init__(self, input_shape, learning_rate=0.001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.pool1 = torch.nn.AvgPool2d(2)
        self.pool2 = torch.nn.AvgPool2d(2)
        self.pool3 = torch.nn.AvgPool2d(2)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 128)
        self.fc2 = nn.Linear(128, 30)

    # returns the size of the output tensor going into Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = nn.Dropout(p=0.3)(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = nn.Dropout(p=0.3)(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = nn.Dropout(p=0.3)(x)

        return x

    # will be used during inference
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(p=0.2)(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x.float())
        loss = F.mse_loss(y_pred, y)

        # tensorboard
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x.float())
        loss = F.mse_loss(y_pred, y)

        # tensorboard
        self.logger.experiment.add_scalar("Loss/Val", loss, self.current_epoch)
        # callback
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x.float())
        loss = F.mse_loss(y_pred, y)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
