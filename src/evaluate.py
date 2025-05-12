import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_NAME = 'D:/Approlabs/models/best_model'
DATA_PATH = 'D:/Approlabs/processed_data'
BATCH_SIZE = 32  # Reduced batch size
MAX_LENGTH = 64  # Reduced sequence length
OUTPUT_DIR = 'D:/Approlabs/evaluation_results'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Move model to CPU if GPU memory is an issue
device = torch.device('cpu')
model = model.to(device)

# Load Test Data
print("Loading test data...")
test_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

def preprocess_texts(texts, tokenizer, max_length=MAX_LENGTH):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

def process_batch(model, inputs, device):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return predictions, probabilities

# Process data in batches
print("Processing data in batches...")
all_predictions = []
all_probabilities = []
all_labels = []

for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch_texts = test_data['text'].iloc[i:i+BATCH_SIZE]
    batch_labels = test_data['label'].iloc[i:i+BATCH_SIZE]
    
    # Preprocess batch
    batch_inputs = preprocess_texts(batch_texts, tokenizer)
    
    # Process batch
    batch_predictions, batch_probabilities = process_batch(model, batch_inputs, device)
    
    # Store results
    all_predictions.extend(batch_predictions)
    all_probabilities.extend(batch_probabilities)
    all_labels.extend(batch_labels)
    
    # Clear memory
    del batch_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)
all_labels = np.array(all_labels)

# Evaluation Metrics
print("\nClassification Report:")
report = classification_report(all_labels, all_predictions)
print(report)

# Save classification report
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(all_labels), yticklabels=set(all_labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# Precision-Recall Curve
print("Generating precision-recall curve...")
precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='b')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'))
plt.close()

# ROC Curve
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
plt.close()

print(f"\nEvaluation results have been saved to: {OUTPUT_DIR}")
