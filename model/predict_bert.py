from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased")

def predict(text):
    result = classifier(text[:512])[0]

    return {
        "label": result["label"],
        "confidence": float(result["score"])
    }

# test
while True:
    text = input("Enter text: ")
    print(predict(text))