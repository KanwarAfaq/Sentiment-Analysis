
```markdown

## 📌 Project Title
**A Controlled Four-Way Transformer Benchmark for Binary Urdu Sentiment Classification on UCSA-20**

## 📖 Description
This repository contains the complete official implementation, preprocessing scripts, and evaluation pipeline for our research on **Urdu text sentiment analysis**. The goal of the project is to systematically compare the performance of four state-of-the-art multilingual pre-trained transformer models on a low-resource language under strictly controlled hyperparameters.

The models evaluated in this project include:
- **BERT-multilingual-cased**
- **BERT-multilingual-uncased**
- **DistilBERT-multilingual-cased**
- **XLM-RoBERTa-base**

---

## 📂 Repository Structure
```text
Sentiment-Analysis/
│
├── main_pipeline.py           # The unified master script for Phase 1 & Phase 2 evaluation
├── requirements.txt           # Exact Python environment dependencies
├── stopwords.txt              # Custom Urdu stop-word dictionary 
│
├── train_dataset.csv          # UCSA-20 training split
├── test_dataset.csv           # UCSA-20 test split
└── README.md

```

---

## 📊 Dataset Information

### Dataset Name

**Urdu Corpus for Sentiment Analysis (UCSA-20)**

### Description

* Total reviews: **9,601**
* Positive samples: **4,843**
* Negative samples: **4,758**
* Language: **Urdu**
* Domain coverage: Politics, movies, TV dramas, sports, and consumer products

### Files Required

* `train_dataset.csv`
* `test_dataset.csv`
* `stopwords.txt` 

---

## ▶️ Usage Instructions

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/KanwarAfaq/Sentiment-Analysis.git](https://github.com/KanwarAfaq/Sentiment-Analysis.git)
cd Sentiment-Analysis

```

### 2️⃣ Install Dependencies

To ensure strict mathematical reproducibility (as reported in the manuscript), install the exact locked dependencies:

```bash
pip install -r requirements.txt

```

### 3️⃣ Execute the Pipeline

Run the master script.

```bash
python main_pipeline.py

```

##  Methodology 

1. **Data Preprocessing:** Missing-value handling, deduplication, and Urdu text normalization (diacritical mark stripping). Stop words are **retained** in the headline configuration as they carry critical sentiment markers like negation.
2. **Tokenization:** WordPiece for BERT/DistilBERT; SentencePiece for XLM-RoBERTa.
3. **Hyperparameters:** - Learning rate: **1e-5**
* Batch size: **16**
* Epochs: **5**
* Max sequence length: **128**
* Random Seed: **42**

---

## 📈 Results Summary

**XLM-RoBERTa-base** achieved the strongest test performance under the controlled baseline:

* **Test Accuracy:** **83.65%**
* **Macro F1-score:** **83.65%**

The results demonstrate that XLM-RoBERTa's advanced SentencePiece tokenization and larger multilingual pretraining corpus provide superior generalization for low-resource fusional languages like Urdu.

---

## 📚 Citation

If you utilize this code or methodology in your research, please cite our paper:
(Citation details will be updated upon publication).

### Dataset Reference

* **Urdu Corpus for Sentiment Analysis (UCSA)** – IEEE DataPort
https://ieee-dataport.org/documents/urdu-corpus-sentiment-analysis


---

## 📜 License

This project is intended for **academic and research purposes only**. Please check the UCSA dataset license for redistribution permissions.

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome.

To contribute:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

## 📬 Contact

For questions or collaboration:

* **Kanwar Muhammad Afaq**: [github.com/KanwarAfaq](https://www.google.com/search?q=https://github.com/KanwarAfaq)
* **Chaithra Lokasara Mahadevaswamy**: [github.com/Chaithra-lm](https://github.com/Chaithra-lm)
* Chang Gung University, Taiwan

---

⭐ If you find this repository useful, please consider giving it a star!

```

```
