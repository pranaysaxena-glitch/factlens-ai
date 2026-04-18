from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import wikipedia

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "../model/model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "../model/vectorizer.pkl"), "rb"))

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Input(BaseModel):
    text: str
    context: str = ""   # for chat memory

def get_evidence(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "No reliable information found."

def predict(text, evidence):
    combined = text + " " + evidence
    vec = vectorizer.transform([combined])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]

    label = "Factual" if pred == 1 else "Hallucinated"
    return label, float(prob)

@app.get("/")
def home():
    return {"message": "FactLens API running"}

@app.post("/predict")
def run_prediction(data: Input):
    combined_text = data.text + " " + data.context

    evidence = get_evidence(combined_text)
    label, confidence = predict(combined_text, evidence)

    return {
        "model": "FactLens v1.0",
        "result": label,
        "confidence": round(confidence, 3),
        "evidence": evidence
    }