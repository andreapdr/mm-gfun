import torch
import lightning as L

from transformers import AutoModelForSequenceClassification


class AutoTextClassificationLightningModule(L.LightningModule):
    def __init__(
            self,
            base_model_name: str,
            num_labels: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler = None,
            lr: float = 5e-5
    ):
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        self.lr = lr

        # Setting logger=True causes deadlock error if num_workers > 1!
        self.save_hyperparameters(logger=False)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            )
        return outputs.logits
    
    def encode(self, batch, batch_idx):
        batch, metadata = batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/input_ids.shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("training_loss", loss, prog_bar=True)
        self.log("training_accuracy", accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.encode(batch, batch_idx)
        self.log("validation_loss", loss)
        self.log("validation_accuracy", accuracy)
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
    from functools import partial

    print("Debugging vision_model...")
    model = AutoTextClassificationLightningModule(
        base_model_name="xlm-roberta-base",
        num_labels=3,
        optimizer=torch.optim.AdamW,
        scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=10)
    )
    model.configure_optimizers()
    print(model)
    
    dummy_inputs = model.base_model.dummy_inputs
    dummy_inputs["attention_mask"] = torch.ones_like(dummy_inputs["input_ids"])
    outputs = model(**dummy_inputs)
    print(f"{outputs=}")