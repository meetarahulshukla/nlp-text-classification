
# Natural Language Processing (NLP) Fundamentals: SMS Spam Classification

A clean, step-by-step natural language processing pipeline to benchmark text vectorization strategies using the classic SMS Spam Collection dataset.

## 🎯 Project Overview
This project serves as a foundational baseline for processing text data into numerical features. It extracts insights from short text messages to classify them as **Ham (legitimate)** or **Spam** by directly comparing two core vectorization techniques:
1. **CountVectorizer (Bag of Words)**: Tracks absolute word frequencies.
2. **TF-IDF Vectorizer**: Balances term frequency against inverse document frequency to down-weight universally common terms.

Both feature matrices are validated using a **Multinomial Naive Bayes** classifier.

---

## 📊 Benchmark Results

| Feature Extraction Technique | Validation Accuracy |
| :--- | :--- |
| **CountVectorizer (Bag of Words)** | **98.39%** 🏆 |
| **TF-IDF Vectorizer** | **96.68%** |

### Key Insight
For short-form messaging (SMS), absolute keyword presence (*"FREE"*, *"WIN"*, *"URGENT"*) serves as a stronger indicator of spam. The document-frequency penalisation applied by TF-IDF slightly dampens the classification signals of these critical high-frequency target words.

---

## 📁 Repository Structure

```text
nlp-fundamentals/
│
├── data/
│   └── spam.csv                     # Raw Kaggle SMS dataset
│
├── src/
│   ├── NLP_1.ipynb         # Part 1: Conceptual matrix walkthrough
│   └── NLP_1_2.ipynb         # Part 2: End-to-end spam classification
│
├── .gitignore                       # Skips temporary system & cache files
├── README.md                        # Documentation and project summaries
└── requirements.txt                 # Project environment dependencies
```

---

## 🛠️ Data Handling Nuances
The standard `spam.csv` dataset contains character encoding quirks and structural anomalies. This pipeline explicitly handles those by:
* Using `ISO-8859-1` (`latin-1`) encoding to cleanly parse special currency and text symbols.
* Utilizing explicit positional indexing (`df.iloc[:, :2]`) to cleanly discard corrupted trailing columns native to the raw layout.

---

## ⚙️ Quick Start & Setup

### 1. Clone the repository
```bash
git clone https://github.com
cd nlp-fundamentals
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline
```bash
python src/02_spam_pipeline.py
```
