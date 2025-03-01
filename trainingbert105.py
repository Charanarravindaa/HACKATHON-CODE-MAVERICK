import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainerCallback

# Load Dataset
dataset = load_dataset('csv', data_files={'train': 'train_data.csv'}, split='train')

# Preprocessing function
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    texts = [str(text) if text is not None else "" for text in examples['cleaned_text']]
    labels = examples['Label']
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
    return {**encodings, 'Labels': labels}

dataset = dataset.map(preprocess_function, batched=True)

# Split the data into train and test
train_data, test_data = dataset.train_test_split(test_size=0.05).values()

# Create the PyTorch dataset
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_texts = [str(text) if text is not None else "" for text in train_data['cleaned_text']]
test_texts = [str(text) if text is not None else "" for text in test_data['cleaned_text']]

# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Create the PyTorch dataset
train_dataset = FakeNewsDataset(train_encodings, list(train_data['Label']))
test_dataset = FakeNewsDataset(test_encodings, list(test_data['Label']))

# Define the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    max_steps=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.005,
    logging_dir='./logs',
    logging_steps=5,
    evaluation_strategy="steps",  # Evaluate every X steps
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    eval_steps=5  # Evaluate every 5 steps
)

# Define a Custom Callback to Stop Training if Loss is Below the Threshold
class LossStopCallback(TrainerCallback):
    def __init__(self, loss_threshold=0.5):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, eval_dataloader=None, **kwargs):
        eval_results = kwargs.get("metrics", {})
        loss = eval_results.get("eval_loss", None)  # Ensure the loss is fetched correctly

        if loss is not None:
            print(f"Loss during evaluation: {loss}")
            if loss <= self.loss_threshold:
                print(f"Stopping training as loss reached {loss} which is below the threshold.")
                control.should_terminate = True  # Stop the training if loss is below threshold
        else:
            print("No evaluation loss found.")
        return control

# Instantiate the callback
loss_stop_callback = LossStopCallback(loss_threshold=0.5)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.predictions.argmax(axis=-1), p.label_ids),
        'precision': precision_score(p.predictions.argmax(axis=-1), p.label_ids),
        'recall': recall_score(p.predictions.argmax(axis=-1), p.label_ids),
        'f1': f1_score(p.predictions.argmax(axis=-1), p.label_ids)
    },
    callbacks=[loss_stop_callback]  # Pass the callback to stop the training when loss is below threshold
)

# Train the model
trainer.train()

# Evaluate and save the model
results = trainer.evaluate()
print(f"Evaluation Results: {results}")
torch.save(model.state_dict(), 'distilbert_custom.pth')
