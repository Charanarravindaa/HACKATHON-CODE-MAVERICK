from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.load_state_dict(torch.load('distilbert_custom.pth'))
model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        result = model(**encoding)
    logits = result.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
