import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report

# Load model & tokenizer
model_path = "./results/checkpoint-3840"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define label mapping
labels_map = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

# Load test data
df = pd.read_csv("./data/valid.tsv", sep="\t", header=None)
df = df[[2, 1]]
df.columns = ["text", "label"]
df["label"] = df["label"].map(labels_map)

# Drop rows with NaN labels (unmapped labels)
df = df.dropna()

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Function to get predictions
def predict(texts):
    texts = [str(text) for text in texts]  # Convert all inputs to strings
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return np.argmax(outputs.logits.numpy(), axis=-1)

# Get predictions
predictions = predict(df["text"].tolist())
labels = df["label"].tolist()

# Compute accuracy and classification report
accuracy = accuracy_score(labels, predictions)
report = classification_report(labels, predictions, target_names=list(labels_map.keys()))

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)
