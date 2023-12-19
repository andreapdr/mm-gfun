import torch
import lightning as L

from transformers import AutoModelForImageClassification
from torch.optim import AdamW


class AutoImageClassificationLightningModule(L.LightningModule):
    def __init__(
            self,
            base_model_name: str,
            num_labels: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            lr:float =5e-5,
        ):
        super().__init__()
        self.base_model = AutoModelForImageClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        
        self.save_hyperparameters(logger=False)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        return outputs.logits
    
    def encode(self, batch, batch_idx):
        batch, metadata = batch
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
    
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)
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
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    print("Debugging vision_model...")
    model = AutoImageClassificationLightningModule(
        base_model_name="google/vit-base-patch16-224",
        num_labels=10,
        optimizer=torch.optim.SGD,
        # scheduler=torch.optim.lr_scheduler.ConstantLR
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR
    )
    model.configure_optimizers()