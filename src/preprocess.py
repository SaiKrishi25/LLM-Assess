import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Load datasets
DATA_PATH = 'E:/Approlabs_project/llm-zero-shot-classifiers/dataset'
OUTPUT_PATH = 'E:/Approlabs_project/llm-zero-shot-classifiers/processed_data/'

os.makedirs(OUTPUT_PATH, exist_ok=True)

DATASETS = [
    'Corona_NLP_train.csv',
    'ecommerceDataset.csv',
    'financial_sentiment.csv',
    'sms_spam.csv'
]

# Initialize tokenizer
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

full_data = []

for dataset in DATASETS:
    try:
        # First try with UTF-8
        data = pd.read_csv(os.path.join(DATA_PATH, dataset))
    except UnicodeDecodeError:
        # If UTF-8 fails, try with Latin-1 (which can handle most encodings)
        data = pd.read_csv(os.path.join(DATA_PATH, dataset), encoding='latin-1')
    
    if 'text' in data.columns:
        text_col = 'text'
    elif 'message' in data.columns:
        text_col = 'message'
    else:
        text_col = data.columns[0]
    
    label_col = data.columns[-1]
    data = data[[text_col, label_col]]
    data.columns = ['text', 'label']
    full_data.append(data)

# Combine all datasets
combined_data = pd.concat(full_data).dropna().reset_index(drop=True)

# Split into train and test
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# Tokenize data
def tokenize_data(data):
    return tokenizer(
        list(data['text']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_data.to_csv(os.path.join(OUTPUT_PATH, 'train.csv'), index=False)
test_data.to_csv(os.path.join(OUTPUT_PATH, 'test.csv'), index=False)

print("Data preprocessing completed. Train and test sets are saved in:", OUTPUT_PATH)
