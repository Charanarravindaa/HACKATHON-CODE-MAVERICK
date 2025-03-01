# app/model.py
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class FakeNewsModel:
    def __init__(self, model_path='model/distilbert_custom.pth'):
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction
