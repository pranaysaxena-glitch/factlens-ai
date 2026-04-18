import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Debug: check files in data folder
print("Files in data folder:", os.listdir("../data"))

# Load datasets
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

# Remove nulls
fake = fake.dropna()
true = true.dropna()

# Assign labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

print("Total rows:", len(data))
print("Label distribution:\n", data["label"].value_counts())

# Use text column
data["question"] = "news"
data["answer"] = data["text"]

# Combine
data["text_combined"] = data["question"] + " " + data["answer"]

X = data["text_combined"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")