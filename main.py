import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#latest checkpoint
def get_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(directory) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError("No checkpoints found in the results directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))  # Get highest checkpoint number
    return os.path.join(directory, latest_checkpoint)

# Load trained model
results_dir = "./results"
latest_checkpoint = get_latest_checkpoint(results_dir)

model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Use the original tokenizer
fact_checker = pipeline("text-classification", model=model, tokenizer=tokenizer)

def check_fact(query):
    result = fact_checker(query)[0]
    labels = ["Pants on Fire", "False", "Barely True", "Half True", "Mostly True", "True"]
    return f"Verdict: {labels[int(result['label'].split('_')[-1])]}"

if __name__ == "__main__":
    query = input("Enter a claim to fact-check: ")
    print(check_fact(query))
