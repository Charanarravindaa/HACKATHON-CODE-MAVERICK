import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Example dataset: You should replace this with your actual data
texts = ["Sample news article text", "Another sample news article"]
labels = [0, 1]  # Example labels, 0 for real, 1 for fake

# Tokenize the dataset
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Track loss
loss_values = []

# Training loop
for epoch in range(10):  # Set your number of epochs
    model.train()  # Set model to training mode
    epoch_loss = 0  # Initialize loss for the epoch

    for batch in train_loader:
        input_ids, attention_mask, labels = batch

        # Forward pass
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Extract the loss from the output

        # Backward pass and optimization
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

        epoch_loss += loss.item()  # Accumulate loss for this batch

    # Average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_loss)  # Track the loss

    print(f'Epoch {epoch+1}: Loss = {avg_loss}')

# Visualize the loss graph
plt.plot(range(1, 11), loss_values)  # Epochs vs loss values
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BERT Loss During Training')
plt.show()
