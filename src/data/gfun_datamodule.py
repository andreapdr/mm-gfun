from typing import Optional, List
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers import PreTrainedTokenizerBase

import torch
import lightning as L
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_from_disk, Image

from pathlib import Path
from tqdm import tqdm


class MultiModalProcessor:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            image_processor: AutoProcessor,
            skip_text: bool = False,
            skip_image: bool = False,
            max_length: Optional[int] = None,
            pad_to_multiple_of: int = 8,
            padding: str = "longest",
            truncation: str = "longest_first",
            return_tensors: str = "pt",
            ):
        assert padding in ["longest", "max_length", "do_not_pad"]
        assert truncation in ["longest_first", "do_not_truncate"]
        assert return_tensors in ["pt", "np", "tf"]

        self.pretrained_tokenizer = tokenizer
        self.image_processor = image_processor
        self.skip_text = skip_text
        self.skip_image = skip_image

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        features.update(
            self.pretrained_tokenizer(
                features["text"],
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors
            )
        )
        if not self.skip_image:
            features["pixel_values"] = torch.tensor(
                np.array(
                    [self.image_processor(img.convert("RGB"))["pixel_values"][0] for img in features["image"]]
                    ),
                )

        features["labels"] = torch.tensor(features["labels"])
        return features


@dataclass
class MultiModalCollator:
    return_columns: List
    metadata: Optional[List] = None

    def __call__(self, features):
        features = {key: [example[key] for example in features] for key in features[0].keys()}
        batch = {k: torch.stack(features[k]) for k in self.return_columns}
        batch_metadata = {k: features[k] for k in self.metadata}
        return batch, batch_metadata 


class MMgFunDataModule(L.LightningDataModule):

    RETURN_COLUMNS_MAPPER = {"glami": ["input_ids", "attention_mask", "pixel_values", "labels"],}
    METADATA_COLUMNS = {"glami": ["geo", "item_id", "image_id", "category_name", "label_source", "name", "description"],}

    def __init__(
            self,
            dataset: str,
            dataset_dir: str,
            num_workers: int,
            batch_size: int,
            text_model_name: str,
            vision_model_name: str,
            accelerator: Optional[str],
            max_rows : Optional[int] = -1,
            max_length: Optional[int] = None,
            skip_images: bool = False,
            load_from_cache: bool = True,
            num_labels: Optional[int] = None,
            ):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = Path(dataset_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.skip_images = skip_images
        self.max_length = max_length
        self.num_labels = num_labels

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.multimodal_processor = MultiModalProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_model_name),
            image_processor=AutoProcessor.from_pretrained(vision_model_name),
            skip_image=self.skip_images,
            max_length=self.max_length,
        )
        
        return_columns = self.RETURN_COLUMNS_MAPPER[self.dataset]
        if self.skip_images:
            return_columns.remove("pixel_values")

        self.collator = MultiModalCollator(
            return_columns=return_columns,
            metadata=self.METADATA_COLUMNS[self.dataset]
            )
        
        self.max_rows = max_rows
        self.load_from_cache = load_from_cache 
    
    def setup(self, stage: Optional[str] = None) -> None:
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            datadir = str(self.dataset_dir.expanduser())

            dataset = load_from_disk(datadir)
            dataset.pop("test")
            
            if self.max_rows != -1:
                dataset["train"] = dataset["train"].select(range(self.max_rows))
                dataset["validation"] = dataset["validation"].select(range(self.max_rows))
            
            dataset = dataset.cast_column("image", Image())
            
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]

            self.train_dataset.set_transform(self.multimodal_processor)
            self.val_dataset.set_transform(self.multimodal_processor)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            drop_last=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def num_training_samples(self) -> int:
        return len(self.train_dataset)
    
    def num_validation_samples(self) -> int:
        return len(self.val_dataset)
    
    def num_test_samples(self) -> int:
        return len(self.test_dataset)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dataset=}, {self.num_workers=}, {self.pin_memory=}, {self.batch_size=}, {self.num_training_samples()=})"


def main():
    """Debug MMgFunDataModule"""
    VISION_MODEL_NAME = "google/vit-base-patch16-224"
    TEXT_MODEL_NAME = "xlm-roberta-base"

    dataset = MMgFunDataModule(
        dataset="glami",
        dataset_dir="data/GLAMI-1M-dataset",
        accelerator="gpu",
        num_workers=2,
        batch_size=64,
        max_rows=-1,
        skip_images=True,
        max_length=32,
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        )
    dataset.setup()
    tr_dataloader = dataset.train_dataloader()
    print(dataset)

    print("\nIterating over training dataloader...")
    for i, (batch, meta) in enumerate(tqdm(tr_dataloader)):
        pass
    
    print("\nLast batch shapes:")
    for k, v in batch.items():
        print(k, v.shape)

    print("\nLast batch metadata")
    for k, v in meta.items():
        print(k, v)

if __name__ == "__main__":
    main()