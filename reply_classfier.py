import os
import re
import pandas as pd
import numpy as np
import nltk
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

# --- 0. SETTINGS & PREPARATION ---
DATASET_PATH = "reply_classification_dataset.csv"
os.environ["WANDB_DISABLED"] = "true"

# --- 1. DATA LOADING AND PREPROCESSING ---
print("--- 1. Loading and Preprocessing Data ---")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at '{DATASET_PATH}'")
    exit()

df.columns = [col.strip().lower() for col in df.columns]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['label'] = df['label'].str.lower()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df['cleaned_reply'] = df['reply'].apply(preprocess_text)
print("✅ Data loaded and cleaned successfully.\n")


# --- 1B. Creating a Holdout Test Set (Crucial for preventing data leakage) ---
print("--- 1B. Creating a Holdout Test Set ---")
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Original training set size: {len(df_train)}")
print(f"Holdout test set size: {len(df_test)}\n")


# --- 1C. CONTROLLED DATA AUGMENTATION (ON TRAINING DATA ONLY) ---
print("--- 1C. Augmenting ONLY the Training Data ---")
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

aug = naw.SynonymAug(aug_src='wordnet')
augmented_data = []
for index, row in df_train.iterrows(): # We only iterate over the training data
    original_text = row['cleaned_reply']
    label = row['label']
    # Only augment sentences with more than 3 words to preserve short, clear signals
    if len(original_text.split()) > 3:
        augmented_text_result = aug.augment(original_text)
        # Ensure the augmented output is a string
        if isinstance(augmented_text_result, list):
            augmented_text = augmented_text_result[0]
        else:
            augmented_text = augmented_text_result
        augmented_data.append({'cleaned_reply': augmented_text, 'label': label})
        
df_augmented_new = pd.DataFrame(augmented_data)
# The final training set is the original training set plus the new augmented data
df_train_final = pd.concat([df_train, df_augmented_new], ignore_index=True)
print(f"Final training set size (with augmentations): {len(df_train_final)}")
print("✅ Controlled data augmentation complete.\n")


# --- 2. BASELINE MODEL (LOGISTIC REGRESSION) ---
print("--- 2. Training Baseline Model (Logistic Regression) ---")
le = LabelEncoder()
# Fit the encoder on the training data and transform both sets
df_train_final['label_encoded'] = le.fit_transform(df_train_final['label'])
df_test['label_encoded'] = le.transform(df_test['label']) # Use the same encoder for the test set

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(df_train_final['cleaned_reply'])
X_test_tfidf = vectorizer.transform(df_test['cleaned_reply'])

lr_model = LogisticRegression(random_state=42, class_weight='balanced')
lr_model.fit(X_train_tfidf, df_train_final['label_encoded'])

# Evaluate on the untouched holdout test set
y_pred = lr_model.predict(X_test_tfidf)
baseline_accuracy = accuracy_score(df_test['label_encoded'], y_pred)
baseline_f1 = f1_score(df_test['label_encoded'], y_pred, average='weighted')
print("\n--- 🤖 Baseline Model Evaluation ---")
print(f"Accuracy on TRUE Test Set: {baseline_accuracy:.4f}")
print(f"Weighted F1 Score on TRUE Test Set: {baseline_f1:.4f}")
print("----------------------------------\n")


# --- 3. TRANSFORMER MODEL (DISTILBERT) ---
print("--- 3. Preparing Data for Transformer Model ---")
# Create datasets from our final training and untouched test frames
train_dataset = Dataset.from_pandas(df_train_final)
eval_dataset = Dataset.from_pandas(df_test)

labels_list = sorted(df_train_final['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels_list)}

class_label_feature = ClassLabel(names=labels_list)
train_dataset = train_dataset.cast_column("label", class_label_feature)
eval_dataset = eval_dataset.cast_column("label", class_label_feature)
train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset = eval_dataset.rename_column("label", "labels")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['cleaned_reply'], padding='max_length', truncation=True)
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(labels_list),
    id2label={i: label for label, i in label2id.items()},
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2, # Using 2 epochs as a good starting point to prevent overfitting
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch", # Correct for your local environment
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("--- 4. Fine-Tuning Transformer Model (DistilBERT) ---")
trainer.train()
print("🎉 Fine-tuning complete!\n")

print("--- 🚀 Transformer Model Evaluation ---")
# The final evaluation is on the holdout eval_dataset
eval_results = trainer.evaluate()
transformer_accuracy = eval_results['eval_accuracy']
transformer_f1 = eval_results['eval_f1']
print(f"Accuracy on TRUE Test Set: {transformer_accuracy:.4f}")
print(f"Weighted F1 Score on TRUE Test Set: {transformer_f1:.4f}")
print("-------------------------------------\n")


# --- 5. FINAL COMPARISON ---
print("--- 🏁 Final Model Comparison ---")
print(f"Baseline (Logistic Regression)  | Accuracy: {baseline_accuracy:.4f} | F1 Score: {baseline_f1:.4f}")
print(f"Transformer (DistilBERT)        | Accuracy: {transformer_accuracy:.4f} | F1 Score: {transformer_f1:.4f}")
print("---------------------------------")


# --- 6. SAVE THE BEST MODEL FOR DEPLOYMENT ---
print("--- 6. Saving Model for API ---")
output_dir = "./reply_classifier_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Best model saved to '{output_dir}'")