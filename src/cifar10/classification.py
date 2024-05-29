import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import (MetricCollection, Accuracy, Precision,
                          Recall, F1Score, JaccardIndex, CohenKappa, ConfusionMatrix)


__all__ = ["ClassificationModule"]


class ClassificationModule(LightningModule):
    def __init__(self, num_classes: int):
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.val__metric = Accuracy(task="multiclass", average="micro", num_classes=num_classes)

    def training_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="train_loss", value=loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="val_loss", value=loss, on_epoch=True, sync_dist=True)
        self.val_global_metric.update(y_hat, y)
        self.val_classes_metric.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        global_metric_value = self.val_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.val_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.val_global_metric.reset()
        self.val_classes_metric.reset()

    def test_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        self.test_global_metric.update(y_hat, y)
        self.test_classes_metric.update(y_hat, y)

    def on_test_epoch_end(self):
        global_metric_value = self.test_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.test_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.test_global_metric.reset()
        self.test_classes_metric.reset()


