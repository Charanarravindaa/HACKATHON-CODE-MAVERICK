import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load your dataset
df = pd.read_csv('train_data.csv')

# Preprocessing: We will use the 'title' and 'cleaned_text' columns to combine them for training
df['text'] = df['title'] + " " + df['cleaned_text']  # Combine text columns

# Convert the dataframe to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load GPT-2 tokenizer
model_name = "gpt2"  # Or any other GPT model you're fine-tuning
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Ensure that the tokenizer is applied to a list of strings
    return tokenizer(examples['text'], padding=True, truncation=True, return_tensors="pt")

# Apply the tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set the format to 'torch' for PyTorch compatibility
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "Label"])

# Split dataset into train and validation sets (optional)
train_dataset = tokenized_datasets.train_test_split(test_size=0.1)["train"]
eval_dataset = tokenized_datasets.train_test_split(test_size=0.1)["test"]

# Load GPT-2 model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels based on your dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",  # Directory to save the model
    num_train_epochs=3,             # Number of epochs to train
    per_device_train_batch_size=4,  # Batch size
    gradient_accumulation_steps=8,  # Gradient Accumulation (optional)
    save_steps=1000,                # Save checkpoint every 1000 steps
    logging_dir="./logs",           # Directory for logs
    logging_steps=500,              # Log every 500 steps
    remove_unused_columns=False     # Prevent issues with data columns not used
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Set evaluation dataset
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Optionally, save the fine-tuned model
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")
