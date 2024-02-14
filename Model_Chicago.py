#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:21:12 2024

@author: ryanhathaway
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd



df = pd.read_csv('/Users/ryanhathaway/Desktop/Project/Anno_Chicago.csv')
#smalltest

small_test = df.sample(n=100)

# Define your sentences and labels
sentences = small_test['Sentence'].tolist()



labels = small_test['Definition?'].tolist()  #  # Labels (1 for definition , 0 for negative)


# Load pre-trained Sentence-BERT model and tokenizer
model_name = "sentence-transformers/paraphrase-mpnet-base-v2"  # Optimal model for long English sentences


#Path to the directory containing the downloaded model files
model_directory = "/Users/ryanhathaway/Desktop/Project/Model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_directory, num_labels=2)  # 2 for binary classification

tokenized_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Extract input_ids and attention_mask from the tokenized inputs
input_ids = tokenized_input['input_ids']
attention_mask = tokenized_input['attention_mask']

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels, train_mask, val_mask = train_test_split(
    input_ids, labels, attention_mask, test_size=0.2, random_state=42
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/Users/ryanhathaway/Desktop/Trained_models/Chicago_Trained/output',  # Specify the directory where checkpoints and logs will be saved
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=4,
   # logging_dir= '/Users/ryanhathaway/Desktop/Trained_models/Chicago_Trained/logs',
    evaluation_strategy="epoch",
    
)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        return {
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'labels': label
        }

train_dataset = CustomDataset(train_inputs, train_mask, train_labels)
val_dataset = CustomDataset(val_inputs, val_mask, val_labels)


# Define the compute_metrics function to include accuracy, precision, recall, and F1-score
def compute_metrics(pred):
    labels = val_labels
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = np.mean(preds == labels)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Define Trainer with updated compute_metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# Train the model
print("Training started...")
trainer.train()
    



# Evaluate the model on the validation dataset
print("Evaluation started...")
results = trainer.evaluate(eval_dataset=val_dataset)
print("Evaluation completed!")

print("Evaluation results:", results)


#Save the model if needed
trainer.save_model("sbert_classification_model")
