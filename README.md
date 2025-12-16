# Sentiment Analysis for Urdu using Multilingual Transformer Models

## 📌 Project Title
**Comparative Analysis of Multilingual Transformer Models for Urdu Sentiment Analysis**

---

## 📖 Description
This repository contains the complete implementation of a research project that performs **sentiment analysis on Urdu text** using state-of-the-art **multilingual transformer models**. The goal of the project is to systematically compare the performance of different multilingual pre-trained transformer models on a low-resource language (Urdu) using the **Urdu Corpus for Sentiment Analysis (UCSA)**.

The models implemented and evaluated in this project include:
- **BERT-multilingual-cased**
- **BERT-multilingual-uncased**
- **DistilBERT-multilingual-cased**
- **XLM-RoBERTa-base**

The repository provides preprocessing scripts, training notebooks, and evaluation results used in the accompanying research paper.

---

## 📂 Repository Structure
```
Sentiment-Analysis/
│
├── BERT_multilingual_cased Training.ipynb
├── bert-multilingual-uncased Training.ipynb
├── distil-bert training.ipynb
├── xlm-roberta-base training.ipynb
│
├── text-preprocessing for bert and models.ipynb
│
├── train_dataset.csv
├── test_dataset.csv
└── README.md
```

---

## 📊 Dataset Information

### Dataset Name
**Urdu Corpus for Sentiment Analysis (UCSA)**

### Description
- Total reviews: **9,601**
- Positive samples: **4,843**
- Negative samples: **4,758**
- Language: **Urdu**
- Domain coverage: Politics, movies, TV dramas, sports, and consumer products

The dataset is **balanced**, making it suitable for binary sentiment classification tasks.

### Files Used
- `train_dataset.csv` – Training data (80%)
- `test_dataset.csv` – Test data (20%)

Each dataset contains:
- `text` – Urdu review text
- `label` – Sentiment label (positive / negative)

---

## 💻 Code Information

### Preprocessing Notebook
- **`text-preprocessing for bert and models.ipynb`**
  - Removes HTML tags and non-standard characters
  - Normalizes Urdu characters
  - Removes Urdu stopwords (NLTK-based)
  - Handles missing values and duplicates

### Training Notebooks
Each model has a dedicated Jupyter Notebook for fine-tuning and evaluation:

- **`BERT_multilingual_cased Training.ipynb`**
- **`bert-multilingual-uncased Training.ipynb`**
- **`distil-bert training.ipynb`**
- **`xlm-roberta-base training.ipynb`**

Each notebook includes:
- Model loading using Hugging Face Transformers
- Tokenization (WordPiece / SentencePiece)
- Training loop with AdamW optimizer
- Validation and testing
- Performance metrics (Accuracy, Precision, Recall, F1-score)
- Loss and accuracy plots

---

## ▶️ Usage Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/KanwarAfaq/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2️⃣ Install Dependencies
Ensure Python ≥ 3.8 is installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Preprocess the Data
Run the preprocessing notebook:
```bash
text-preprocessing for bert and models.ipynb
```

### 4️⃣ Train Models
Run any of the following notebooks based on the model you want to train:
- `BERT_multilingual_cased Training.ipynb`
- `bert-multilingual-uncased Training.ipynb`
- `distil-bert training.ipynb`
- `xlm-roberta-base training.ipynb`

### 5️⃣ Evaluate Results
Each notebook outputs:
- Training & validation loss
- Training & validation accuracy
- Test accuracy, precision, recall, and F1-score

---

## ⚙️ Requirements

Key dependencies used in this project:
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK

Example installation:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn nltk
```

---

## 🧪 Methodology

1. **Data Collection & Annotation**  
   Utilized the publicly available UCSA dataset, manually annotated by Urdu language experts.

2. **Data Preprocessing**  
   Normalization, stopword removal, and duplicate handling specific to Urdu linguistic challenges.

3. **Tokenization**  
   - WordPiece for BERT and DistilBERT
   - SentencePiece for XLM-RoBERTa

4. **Model Fine-tuning**  
   - Learning rate: **1e-5** (optimal)
   - Batch size: **16**
   - Epochs: **5**
   - Max sequence length: **128**

5. **Evaluation Metrics**  
   Accuracy, Precision, Recall, and F1-score

---

## 📈 Results Summary

- **XLM-RoBERTa-base** achieved the best performance:
  - Test Accuracy: **79.51%**
  - F1-score: **79.47%**

- Demonstrates superior generalization for low-resource languages like Urdu.

---

## 📚 Citation
If you use this code or dataset, please cite:

> Kanwar Muhammad Afaq, Chaithra Lokasara Mahadevaswamy, Ammar Amjad, Hsien-Tsung Chang.  
> *Comparative Analysis of Multilingual Transformer Models for Urdu Sentiment Analysis*, 2024.

### Dataset Reference
- **Urdu Corpus for Sentiment Analysis (UCSA)** – IEEE DataPort  
  https://ieee-dataport.org/documents/urdu-corpus-sentiment-analysis

### Related Peer-Reviewed Publications Using UCSA
The UCSA dataset has also been used in the following peer-reviewed research works:

1. **Khan, L., Amjad, A., Ashraf, N., et al.**  
   *Multi-class sentiment analysis of Urdu text using multilingual BERT.*  
   Scientific Reports, 12, 5436 (2022).  
   https://doi.org/10.1038/s41598-022-09381-9

2. **Khan, L., Qazi, A., Chang, H.-T., et al.**  
   *Empowering Urdu sentiment analysis: an attention-based stacked CNN-Bi-LSTM DNN with multilingual BERT.*  
   Complex & Intelligent Systems, 11, 10 (2025).  
   https://doi.org/10.1007/s40747-024-01631-9

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
- **Kanwar Muhammad Afaq**: github.com/KanwarAfaq
- **Chaithra Lokasara Mahadevaswamy**: github.com/Chaithra-lm
- Chang Gung University, Taiwan

---

⭐ If you find this repository useful, please consider giving it a star!

