import pandas as pd  # type: ignore
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

#loading data
df = pd.read_csv('./data/train.tsv', sep='\t', header=None)
df = df[[1, 2]]
df.columns = ["label", "claim"]

#eval data
eval_df = pd.read_csv('./data/valid.tsv', sep='\t', header=None)
eval_df = eval_df[[1, 2]]
eval_df.columns = ["label", "claim"]

# Define mapping for labels to numbers
label_mapping = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

# Convert labels into numbers
df["label"] = df["label"].map(label_mapping)
eval_df["label"] = eval_df["label"].map(label_mapping)

#tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokens = tokenizer(df["claim"].to_list(), truncation=True, padding=True, max_length=512)
eval_tokens = tokenizer(eval_df["claim"].to_list(), truncation=True, padding=True, max_length=512)

#dataset creation
class FactCheckerDS(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[index])  # Add labels
        return item

dataset = FactCheckerDS(tokens, df["label"].to_list())
eval_dataset = FactCheckerDS(eval_tokens, eval_df["label"].to_list())

#train the pretrained model on our data
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset
)

trainer.train()

        