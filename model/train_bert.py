import pandas as pd
from transformers import pipeline

# Load data
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true]).dropna()

texts = data["text"].tolist()
labels = data["label"].tolist()

# Use pretrained model (no heavy training needed)
classifier = pipeline("text-classification", model="distilbert-base-uncased")

print("BERT model ready!")