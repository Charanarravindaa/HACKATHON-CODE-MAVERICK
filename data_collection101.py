##Ticket 101: Fake News Detection Model - Data Collection and Preprocessing
##Objective: Collect and preprocess datasets for training the fake news detection model.
#Steps:
#Dataset Collection:


#Collect diverse datasets (e.g., Kaggle’s fake news dataset, LIAR dataset, etc.).
#Ensure data includes both real and fake news articles from multiple domains (politics, technology, etc.).
#Data Preprocessing:


#Clean the text data (remove unnecessary characters, tokenize, and handle special characters).
#Normalize text using NLTK or spaCy to remove stop words and perform stemming/lemmatization.
##Convert the text into vectors using TF-IDF or word embeddings.
#Split Data:


#Split data into training and testing sets (80/20 split).
#Deliverables:
 #✅ Preprocessed dataset for training.
 #✅ Cleaned text data ready for model training.


import pandas as pd
import re

from sklearn.model_selection import train_test_split

# Step 1: Load the data
fake_df = pd.read_csv('Fake.csv')
truth_df = pd.read_csv('True.csv')

# Step 2: Label the data
fake_df['Label'] = 0  # Fake news as 0
truth_df['Label'] = 1  # Real news as 1

# Combine the datasets
df = pd.concat([fake_df, truth_df], axis=0)

# Step 3: Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Step 4: Define a function to clean the text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase and strip extra spaces
    text = text.strip().lower()
    return text

# Apply the text cleaning function to the 'Text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Step 5: Create the final cleaned dataset with the relevant columns
cleaned_df = df[['title', 'cleaned_text', 'subject', 'date', 'Label']]

# Step 6: Export the cleaned dataset to a new CSV file
cleaned_df.to_csv('cleaned_news_data.csv', index=False)

print("Cleaned dataset has been saved as 'cleaned_news_data.csv'")
cleaned_df = pd.read_csv('cleaned_news_data.csv')

df = pd.read_csv('cleaned_news_data.csv')

# Step 2: Split the data into 80% training and 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Save the split datasets into separate CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("Data has been split into training and testing sets.")