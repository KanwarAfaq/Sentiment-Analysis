# Multilingual Transformers for Low-Resource Urdu Sentiment Analysis

## 📌 Project Overview
This repository contains the official implementation of our research benchmarking state-of-the-art multilingual transformer models for **Urdu Sentiment Analysis**. The study provides a controlled, four-way comparison on the low-resource **Urdu Corpus for Sentiment Analysis (UCSA-20)**.

Our methodology emphasizes strict cross-validation, hyperparameter parity, automated LaTeX table generation, and controlled ablation studies regarding Urdu stop-word retention.

## 🏆 Key Findings
In a controlled 5-fold stratified cross-validation setup, **XLM-RoBERTa-base** outperformed competing BERT architectures:
* **Test Accuracy:** 83.65%
* **Macro F1:** 83.65%
* **Insight:** Retaining Urdu stop-words (such as negation markers) preserves vital contextual sentiment cues, yielding superior performance compared to traditional stop-word removal.

## 📦 Models Evaluated
1. `xlm-roberta-base` (Winner)
2. `bert-base-multilingual-cased` 
3. `bert-base-multilingual-uncased`
4. `distilbert-base-multilingual-cased`

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Environment Setup
Clone the repository and install the dependencies from `requirements.txt`. An NVIDIA GPU with at least 16GB VRAM (e.g., Tesla T4) is highly recommended.
```bash
git clone [https://github.com/KanwarAfaq/Sentiment-Analysis.git](https://github.com/KanwarAfaq/Sentiment-Analysis.git)
cd Sentiment-Analysis
pip install -r requirements.txt

## 📬 Contact
For questions or collaboration:
- **Kanwar Muhammad Afaq**: github.com/KanwarAfaq
- **Chaithra Lokasara Mahadevaswamy**: github.com/Chaithra-lm
- Chang Gung University, Taiwan

---

⭐ If you find this repository useful, please consider giving it a star!

