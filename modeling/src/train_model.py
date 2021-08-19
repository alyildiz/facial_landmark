import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(model, module, max_epochs):
    logger = TensorBoardLogger("/workdir/tensorboard_logs", name="BasicCNNModel")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, mode="min")
    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=torch.cuda.device_count(),
        log_every_n_steps=5,
        logger=logger,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, module)
