import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Set the path for the data
DATA_FILE = 'depression_dataset_reddit_cleaned.csv'

# Check if the data is available locally
if not os.path.exists(DATA_FILE):
    print(f"Error: The file '{DATA_FILE}' was not found.")
    print("Please upload the CSV file to your environment and try again.")
    # Exit the script if data cannot be loaded
    exit()
else:
    print("Dataset found. Skipping download.")

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Display the first few rows to confirm loading
print("Original DataFrame head:")
print(df.head())
print("\nDataFrame Info:")
df.info()

# --- Data Cleaning ---
def clean_text(text):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and special characters
    text = re.sub(r'\@\w+|\#|[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Note: The provided dataset is already clean, but it's good practice to keep the function
df['clean_text'] = df['clean_text'].apply(clean_text)

print("\nCleaned DataFrame head:")
print(df.head())

# Drop rows with NaN or empty text after cleaning
df.dropna(subset=['clean_text'], inplace=True)
df = df[df['clean_text'].str.strip() != '']

# --- Tokenization and Padding ---
# Parameters for tokenization and padding
VOCAB_SIZE = 10000
MAX_LEN = 100

# Initialize the Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<oov>")
tokenizer.fit_on_texts(df['clean_text'])
word_index = tokenizer.word_index

print(f"\nFound {len(word_index)} unique tokens.")

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(df['clean_text'])

# Pad sequences to a fixed length
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# --- Save Preprocessed Data and Tokenizer ---
X = np.array(padded_sequences)
y = np.array(df['is_depression']) # Note: The column name is 'is_depression' in this dataset

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Save the preprocessed data and tokenizer for later use
np.savez('processed_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nData preprocessing complete. Files 'processed_data.npz' and 'tokenizer.pkl' saved.")
