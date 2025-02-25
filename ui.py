import streamlit as st # type: ignore
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#latest checkpoint
def get_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(directory) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError("No checkpoints found in the results directory.")
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(directory, latest_checkpoint)

# Load trained model
results_dir = "./results"
latest_checkpoint = get_latest_checkpoint(results_dir)

model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
fact_checker = pipeline("text-classification", model=model, tokenizer=tokenizer)

def check_fact(query):
    result = fact_checker(query)[0]
    labels = ["Pants on Fire", "False", "Barely True", "Half True", "Mostly True", "True"]
    return f"Verdict: {labels[int(result['label'].split('_')[-1])]}"

# Streamlit UI
st.title("Fact Checker")
st.write("Enter a claim, and the model will classify its truthfulness.")

query = st.text_input("Enter a claim:")
if st.button("Check Fact"):
    if query:
        verdict = check_fact(query)
        st.success(verdict)
    else:
        st.warning("Please enter a claim to fact-check.")
