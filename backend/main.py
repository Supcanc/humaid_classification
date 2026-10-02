from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_CHECKPOINT_PATH, ID2LABEL
import torch

app = FastAPI()

class TextForClassification(BaseModel):
    string: str

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)

@app.post('/predict/')
def predict(text: TextForClassification):
    model_inputs = tokenizer(text.string, padding=True, truncation=True, return_tensors='pt')
    model_outputs = model(**model_inputs)

    logits = model_outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    labels = [ID2LABEL[pred_id.item()] for pred_id in preds]

    return {
        'class_indices': preds.tolist(),
        'class_labels': labels,
        'classes_probs': probs.tolist()
    }