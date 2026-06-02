# ==============================================================================
# CELL 1: SETUP & INSTALLATION
# ==============================================================================
# Pinning exact versions to match manuscript claims for reproducibility
!pip install -q transformers==4.41.0 sentencepiece torch==2.2.0

from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# CELL 2: THE MASTER PIPELINE (FULL DATASET)
# ==============================================================================
import numpy as np
import pandas as pd
import torch
import re
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW  # Correct PyTorch AdamW import
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# --- 1. GLOBAL HYPERPARAMETERS ---
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
K_FOLDS = 5

# --- 2. GOOGLE DRIVE PATHS ---
# This is where the code will look for your CSVs and save your PDFs
# Change this to "" if running locally instead of Google Colab
BASE_PATH = "/content/drive/MyDrive/Urdu_Sentiment_Data/"

# --- 3. REPRODUCIBILITY & DEVICE ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
if device.type == 'cpu':
    print("⚠️ WARNING: You are running on CPU. For exact reproduction, an NVIDIA Tesla V100 or T4 GPU is recommended.")

# --- 4. DATA LOADING & PREPROCESSING (FULL DATASET) ---
print("\nLoading and merging full datasets...")
try:
    df_train = pd.read_csv(f'{BASE_PATH}train_dataset.csv')
    df_test = pd.read_csv(f'{BASE_PATH}test_dataset.csv')
except FileNotFoundError:
    raise Exception(f"ERROR: Could not find the CSV files! Make sure your CSVs are uploaded to the correct path.")

# Rename columns to match pipeline logic
df_train = df_train.rename(columns={'Tweets': 'review', 'label': 'sentiment'})
df_test = df_test.rename(columns={'Tweets': 'review', 'label': 'sentiment'})


df_train = df_train.dropna(subset=['review', 'sentiment']).drop_duplicates(subset=['review'])
df_test = df_test.dropna(subset=['review', 'sentiment']).drop_duplicates(subset=['review'])


df_raw = pd.concat([df_train, df_test], ignore_index=True).drop_duplicates(subset=['review'])

print(f"Total unique reviews ready for training: {len(df_raw)}")
print("Data loaded successfully for Phase 1  and Phase 2.")

# Load Custom Urdu Stop Words from Supplementary File
stopword_file_path = f'{BASE_PATH}stopwords.txt'

try:
    with open(stopword_file_path, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace, and ignore empty lines
        urdu_stopwords = set(word.strip() for word in f.readlines() if word.strip())
    print(f"Loaded {len(urdu_stopwords)} stop words from supplementary file.")
except FileNotFoundError:
    raise Exception(f"ERROR: Could not find the stop-words file at {stopword_file_path}. Please upload it to your Drive.")

def clean_and_normalize(text):
    text = re.sub(re.compile('<.*?>'), '', str(text))
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    return text.replace('ۀ', 'ہ').replace('ۂ', 'ہ').replace('ؤ', 'و').replace('ئ', 'ی')

def remove_custom_stopwords(text):
    return " ".join([w for w in text.split() if w not in urdu_stopwords])

# --- 5. DATASET & TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class UrduDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
        
    def __len__(self): 
        return len(self.texts)
        
    def __getitem__(self, idx):
        # Updated Tokenizer call (No encode_plus)
        encoding = self.tokenizer(
            str(self.texts[idx]), 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_loader(texts, labels, b_size):
    return DataLoader(UrduDataset(texts, labels, tokenizer, MAX_LEN), batch_size=b_size, shuffle=True, num_workers=2)

# --- 6. VISUALIZATION FUNCTIONS ---
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, config_name):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r--s', label='Val Loss', linewidth=2)
    ax1.set_title('Loss vs. Epochs', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_xticks(epochs); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r--s', label='Val Accuracy', linewidth=2)
    ax2.set_title('Accuracy vs. Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12); ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xticks(epochs); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filename = f"{BASE_PATH}{config_name.replace(' ', '_')}_learning_curves.pdf"
    plt.savefig(filename, dpi=1500, format='pdf')
    plt.close()

def plot_confusion_matrix(cm, config_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title('Test Confusion Matrix', fontsize=14)
    plt.ylabel('Actual Class', fontsize=12); plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    filename = f"{BASE_PATH}{config_name.replace(' ', '_')}_confusion_matrix.pdf"
    plt.savefig(filename, dpi=1500, format='pdf')
    plt.close()

def generate_cv_table(fold_accuracies, fold_f1s, config_name):
    data = {
        'Fold 1': [fold_accuracies[0], fold_f1s[0]],
        'Fold 2': [fold_accuracies[1], fold_f1s[1]],
        'Fold 3': [fold_accuracies[2], fold_f1s[2]],
        'Fold 4': [fold_accuracies[3], fold_f1s[3]],
        'Fold 5': [fold_accuracies[4], fold_f1s[4]],
        'Mean':   [np.mean(fold_accuracies), np.mean(fold_f1s)],
        'Std':    [np.std(fold_accuracies), np.std(fold_f1s)]
    }
    df = pd.DataFrame(data, index=['Accuracy', 'Macro F1']).round(4)
    print(f"\n--- LaTeX Code for Table ({config_name}) ---")
    print(df.to_latex())

# --- 7. MASTER EXPERIMENT RUNNER ---
def run_experiment(config_name, remove_stopwords=False):
    print(f"\n{'#'*50}\nRUNNING: {config_name}\n{'#'*50}")
    
    # Apply preprocessing to all datasets
    train_df = deepcopy(df_train)
    test_df = deepcopy(df_test)
    cv_df = deepcopy(df_raw)
    
    for df in [train_df, test_df, cv_df]:
        df['review'] = df['review'].apply(clean_and_normalize)
        if remove_stopwords: 
            df['review'] = df['review'].apply(remove_custom_stopwords)

    save_dir = f"{BASE_PATH}best_model_{config_name.replace(' ', '_')}"

    # PHASE 1
    
    print("\n--- PHASE 1: STATIC SPLIT EVALUATION (Headline Metrics) ---")
    
    t_texts, t_labels = train_df['review'].to_numpy(), train_df['sentiment'].to_numpy()
    test_texts, test_labels = test_df['review'].to_numpy(), test_df['sentiment'].to_numpy()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        t_texts, t_labels, test_size=0.20, stratify=t_labels, random_state=RANDOM_SEED
    )
    
    train_loader = create_loader(train_texts, train_labels, BATCH_SIZE)
    val_loader = create_loader(val_texts, val_labels, BATCH_SIZE)
    test_loader = create_loader(test_texts, test_labels, BATCH_SIZE)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_loader)*EPOCHS*WARMUP_RATIO), num_training_steps=len(train_loader)*EPOCHS)
    
    t_losses, v_losses, t_accs, v_accs = [], [], [], []
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses, batch_preds, batch_tgts = [], [], []
        for batch in train_loader:
            input_ids, attention_mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tgts)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            batch_losses.append(outputs.loss.item())
            batch_preds.extend(torch.max(outputs.logits, dim=1)[1].cpu().numpy())
            batch_tgts.extend(tgts.cpu().numpy())
            
        t_losses.append(np.mean(batch_losses))
        t_accs.append(accuracy_score(batch_tgts, batch_preds))

        # Validation pass
        model.eval()
        v_batch_losses, val_preds, val_tgts = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tgts)
                v_batch_losses.append(outputs.loss.item())
                val_preds.extend(torch.max(outputs.logits, dim=1)[1].cpu().numpy())
                val_tgts.extend(tgts.cpu().numpy())
        
        v_losses.append(np.mean(v_batch_losses))
        v_accs.append(accuracy_score(val_tgts, val_preds))
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_losses[-1]:.4f} | Val Accuracy: {v_accs[-1]:.4f}")

    # Final Test pass 
    model.eval()
    test_preds, test_tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            test_preds.extend(torch.max(outputs.logits, dim=1)[1].cpu().numpy())
            test_tgts.extend(batch['labels'].to(device).cpu().numpy())
            
 p1_acc = accuracy_score(test_tgts, test_preds)
    _, _, p1_f1, _ = precision_recall_fscore_support(test_tgts, test_preds, average='macro')
    p1_cm = confusion_matrix(test_tgts, test_preds)
    
    print(f"\n  -> PHASE 1 STATIC TEST RESULTS: Accuracy: {p1_acc:.4f} | Macro F1: {p1_f1:.4f}")
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(test_tgts, test_preds, target_names=['Negative', 'Positive'], digits=4))
    
    # Save graphs and model 
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    plot_learning_curves(t_losses, v_losses, t_accs, v_accs, config_name + "_Phase1")
    plot_confusion_matrix(p1_cm, config_name + "_Phase1")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 2: 
    print("\n--- PHASE 2: 5-FOLD CROSS VALIDATION (Variance Testing) ---")
    
    cv_texts, cv_labels = cv_df['review'].to_numpy(), cv_df['sentiment'].to_numpy()
    fold_accuracies, fold_macro_f1s = [], []
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(cv_texts, cv_labels)):
        print(f"\n--- CV Fold {fold + 1} / {K_FOLDS} ---")
        
        f_test_texts, f_test_labels = cv_texts[test_idx], cv_labels[test_idx]
        f_train_texts, f_val_texts, f_train_labels, f_val_labels = train_test_split(
            cv_texts[train_val_idx], cv_labels[train_val_idx], test_size=0.20, stratify=cv_labels[train_val_idx], random_state=RANDOM_SEED
        )
        
        f_train_loader = create_loader(f_train_texts, f_train_labels, BATCH_SIZE)
        f_val_loader = create_loader(f_val_texts, f_val_labels, BATCH_SIZE)
        f_test_loader = create_loader(f_test_texts, f_test_labels, BATCH_SIZE)
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(f_train_loader)*EPOCHS*WARMUP_RATIO), num_training_steps=len(f_train_loader)*EPOCHS)
        
        for epoch in range(EPOCHS):
            model.train()
            for batch in f_train_loader:
                input_ids, attention_mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=tgts)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Final test evaluation 
        model.eval()
        f_test_preds, f_test_tgts = [], []
        with torch.no_grad():
            for batch in f_test_loader:
                outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                f_test_preds.extend(torch.max(outputs.logits, dim=1)[1].cpu().numpy())
                f_test_tgts.extend(batch['labels'].to(device).cpu().numpy())
                
        fold_acc = accuracy_score(f_test_tgts, f_test_preds)
        _, _, fold_f1, _ = precision_recall_fscore_support(f_test_tgts, f_test_preds, average='macro')
        
        fold_accuracies.append(fold_acc)
        fold_macro_f1s.append(fold_f1)
        print(f"  -> Fold {fold+1} Test Accuracy: {fold_acc:.4f} | Macro F1: {fold_f1:.4f}")

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    generate_cv_table(fold_accuracies, fold_macro_f1s, config_name + "_CV")

# --- 8. EXECUTE BOTH EXPERIMENTS ---
# 1. Headline Configuration (Stop words retained)
run_experiment("Headline Configuration", remove_stopwords=False)

# 2. Ablation Configuration (Stop words removed)
run_experiment("Ablation Configuration", remove_stopwords=True)

print(f"\n✅ All done! Your graphs and model weights have been saved.")
