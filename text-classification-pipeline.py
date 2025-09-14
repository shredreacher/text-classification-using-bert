import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# 1. Load AG News dataset
def load_ag_news():
    ds, info = tfds.load('ag_news_subset', with_info=True, as_supervised=True)
    train_ds, test_ds = ds['train'], ds['test']
    return train_ds, test_ds


# 2. Tokenize and preprocess data
def tokenize_batch(tokenizer, texts, labels):
    tokens = tokenizer(
        list(texts),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokens["labels"] = torch.tensor(labels)
    return tokens


def tfds_to_torch(dataset, tokenizer):
    texts, labels = [], []
    for text, label in tfds.as_numpy(dataset):
        texts.append(text.decode())
        labels.append(label)
    return tokenize_batch(tokenizer, texts, labels)


# 3. PyTorch Dataset wrapper
class AGNewsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def main():
    # Model checkpoint
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load and preprocess data
    train_ds, test_ds = load_ag_news()
    train_encodings = tfds_to_torch(train_ds, tokenizer)
    test_encodings = tfds_to_torch(test_ds, tokenizer)

    train_dataset = AGNewsDataset(train_encodings)
    test_dataset = AGNewsDataset(test_encodings)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        do_train=True,
        do_eval=True,
        eval_steps=100,
        save_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Train and save
    trainer.train()
    model.save_pretrained("./fine-tuned-bert-agnews")
    tokenizer.save_pretrained("./fine-tuned-bert-agnews")


if __name__ == "__main__":
    main()

