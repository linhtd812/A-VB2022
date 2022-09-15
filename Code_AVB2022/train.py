import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from dataset import DataModule
from lightning_module import BaselineLightningModule, Wav2vecWrapper, BaselineLightningModule_Spec
import os

seed = 1024
seed_everything(seed)


@hydra.main(config_path="config", config_name="default")
def train(cfg):
    # loggers
    csvlogger = CSVLogger(save_dir=cfg.log_dir, name="csv")
    tblogger = TensorBoardLogger(save_dir=cfg.log_dir, name="tb")
    loggers = [csvlogger, tblogger]

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log_dir,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        monitor="val_ccc",
        mode="max",
    )
    earlystop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        mode="min",
        check_finite=True,
        stopping_threshold=0.0,
        divergence_threshold=1e5,
    )
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, earlystop_callback, lr_monitor]

    datamodule = DataModule(cfg)

    # # Original
    # lightning_module = BaselineLightningModule(cfg)

    # Original and Mel-Spectrogram
    lightning_module = BaselineLightningModule_Spec(cfg)

    # # Load checkpoint (freeze other parts, only train DINO and classifier)
    # save_path = '/mnt/1TSSD/AVB/ExVo/outputs/2022-09-05/23-58-48/'
    # checkpoint_path = (save_path + "pl_log" +"/epoch=40-step=204918.ckpt")
    # lightning_module = BaselineLightningModule_Spec.load_from_checkpoint(checkpoint_path, cfg=cfg)

    # all_layers = []
    # for k, v in lightning_module.named_parameters():
    #     all_layers.append(k)
    # freeze = [x for x in all_layers if "chain" not in x and "dino" not in x]
    # if any(freeze):
    #     for k, v in lightning_module.named_parameters():
    #         if any(x in k for x in freeze):
    #             # print('freezing %s' % k)
    #             v.requires_grad = False

    # print(lightning_module)
    # lightning_module = Wav2vecWrapper(cfg)

    # trainer = pl.Trainer(**cfg.train.trainer, default_root_dir=hydra.utils.get_original_cwd(),
    #                 logger=loggers, callbacks=callbacks)

    trainer = pl.Trainer(
        **cfg.train.trainer,
        default_root_dir=hydra.utils.get_original_cwd(),
        logger=loggers,
        callbacks=callbacks,
    )
    # trainer.tune(lightning_module, datamodule=datamodule)
    trainer.fit(lightning_module, datamodule=datamodule)
    print(
        f"Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}"
    )


if __name__ == "__main__":
    train()
