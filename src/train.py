import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
import pandas as pd
import os

# Configuration
MODEL_NAME = 'bert-base-uncased'
DATA_PATH = '/content/drive/MyDrive/llm-zero-shot-classifiers/processed_data'
MODEL_SAVE_PATH = '/content/drive/MyDrive/llm-zero-shot-classifiers/models'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load data
train_data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
val_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

# Normalize labels
label_mapping = {label: idx for idx, label in enumerate(sorted(set(train_data['label'])))}
train_data['label'] = train_data['label'].map(label_mapping)
val_data['label'] = val_data['label'].map(label_mapping)
print("Label Mapping:", label_mapping)

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = TextDataset(train_data)
val_dataset = TextDataset(val_data)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_mapping)).to(device)

# Training settings
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    learning_rate=2e-5,  # Lower learning rate
    warmup_steps=100,    # Warmup for stable training
    weight_decay=0.01,   # Regularization
    save_steps=100,      # Save checkpoints frequently
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Save the best model
model.save_pretrained(os.path.join(MODEL_SAVE_PATH, 'best_model'))
tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, 'best_model'))

print("Optimized Training completed. Model saved at:", MODEL_SAVE_PATH)