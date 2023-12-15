import torch.nn as nn
import lightning as L

from typing import Optional
from torch.optim import AdamW

from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification


class BaseLightningModule(L.LightningDataModule):
    def __init__(
            self,
            encoder: Optional[nn.Module] = None,
            decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        pass

    def forward(self):
        raise NotImplementedError
    
    def encode(self):
        raise NotImplementedError
    
    def decode(self):
        raise NotImplementedError
    
    def generate(self):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def configure_optimizer(self):
        raise NotImplementedError


class AutoSequenceClassificationLightningModule(L.LightningModule):
    def __init__(
            self,
            base_model_name: str,
            num_labels: int):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def encode(self, batch, batch_idx):
        batch, metadata = batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids=input_ids, attention_mask=attention_mask)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/input_ids.shape[0]

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


class AutoImageClassificationLightningModule(L.LightningModule):
    def __init__(
            self,
            base_model_name: str,
            num_labels: int):
        super().__init__()
        self.base_model = AutoModelForImageClassification.from_pretrained(
            base_model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        
        self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        return outputs.logits
    
    def encode(self, batch, batch_idx):
        batch, metadata = batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


def main():
    from data import MMgFunDataModule
    VISION_MODEL_NAME = "google/vit-base-patch16-224"
    TEXT_MODEL_NAME = "xlm-roberta-base"
    
    dataset = MMgFunDataModule(
        dataset="glami",
        num_workers=4,
        batch_size=8,
        accelerator="gpu",
        max_rows=-1,
        skip_images=False,
        max_length=32,
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        )
    dataset.setup()
    
    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    num_labels = dataset.num_labels()
    print(f"{num_labels=}")

    model = AutoImageClassificationLightningModule(
        base_model_name=VISION_MODEL_NAME,
        num_labels=num_labels
    )

    # model = AutoSequenceClassificationLightningModule(
    #     base_model_name=TEXT_MODEL_NAME,
    #     num_labels=num_labels
    # )

    trainer = L.Trainer(
        logger=True,
        enable_checkpointing=False,
        limit_train_batches=5000,
        limit_val_batches=500,
        log_every_n_steps=50,
        val_check_interval=250,
        max_epochs=2,
        callbacks=[])
    
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    
if __name__ == "__main__":
    main()