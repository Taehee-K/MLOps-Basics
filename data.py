import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    """Pytorch Lightning DataModule

    1. Download / tokenize / process
    2. Clean and (maybe) save to disk
    3. Load inside Dataset
    4. Apply transformers (rotate, tokenize, etc..)
    5. Wrap inside a DataLoader

    """

    def __init__(
        self,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2",
        batch_size: int = 64,
        max_length: int = 128,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        # called once, on 1 GPU
        cola_dataset = load_dataset("glue", "cola")
        self.train_data = cola_dataset["train"]
        self.val_data = cola_dataset["validation"]

    def tokenize_data(self, example):
        return self.tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def setup(self, stage=None):
        # called on each GPU separately
        # set up only relevant datasets when stage is specified
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"],
                output_all_columns=True,    # keep unformatted columns
            )

    # dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )


if __name__ == "__main__":
    data_model = DataModule(
        model_name="google/bert_uncased_L-2_H-128_A-2",
        batch_size=32,
        max_length=128,
    )
    data_model.prepare_data()
    data_model.setup()
    # Check data shape
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
