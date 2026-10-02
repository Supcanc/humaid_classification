from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
import torch

cur_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(cur_dir)

sys.path.append(parent_dir)

from backend.config import MODEL_CHECKPOINT_PATH, ID2LABEL

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