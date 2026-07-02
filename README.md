# 📩 Spam SMS/Email Classifier

A machine learning project that classifies text messages as **Spam** or **Ham (not spam)**
using **TF-IDF** vectorization and a **Multinomial Naive Bayes** classifier.

Built as a hands-on learning project to understand the end-to-end ML pipeline — from raw text to a working classifier.

---

## 📊 Dataset

- **Source:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 labeled SMS messages
- **Classes:** `ham` (4,825 messages) and `spam` (747 messages) — an imbalanced dataset (~86% ham, ~14% spam)

> Note: `spam.csv` is not included in this repo. Download it from the Kaggle link above and place it in the project folder before running.

---

## 🛠️ Tech Stack

- Python
- pandas — data loading & manipulation
- scikit-learn — TF-IDF vectorization, Naive Bayes model, evaluation metrics
- re (regex) — text cleaning

---

## 🔄 Pipeline

1. **Load data** — read the CSV, keep only the label and message columns
2. **Clean text** — lowercase, strip punctuation/numbers, collapse extra whitespace
3. **Encode labels** — `ham` → 0, `spam` → 1
4. **Train/test split** — 80% train, 20% test
5. **Vectorize text** — TF-IDF converts cleaned messages into numeric feature vectors
6. **Train model** — Multinomial Naive Bayes
7. **Evaluate** — accuracy, precision, recall, confusion matrix

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | 95% |
| Precision (spam) | 100% |
| Recall (spam) | 65% |

**Confusion Matrix:**

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 965 | 0 |
| **Actual Spam** | 53 | 97 |

### Key Insight

The model has **perfect precision** but **moderate recall** on spam — meaning it
never wrongly flags a real message as spam, but it misses about a third of actual spam messages.
This is a direct effect of the **class imbalance** in the dataset (far more ham examples than spam),
which biases the model toward predicting the more common class when uncertain. A good next step
would be exploring class-weighting or resampling techniques to improve spam recall.

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn
python spam_classifier.py
```

Make sure `spam.csv` (downloaded from Kaggle) is in the same folder as the script.

---

## 🔮 Future Improvements

- Handle class imbalance (oversampling/undersampling, class weights)
- Try other models (Logistic Regression, SVM) and compare performance
- Add a simple web interface (Flask/Streamlit) to test messages live
- Deploy as an API

---

## 👤 Author

**Rishabh Dohale**
B.Tech AIML, G H Raisoni College, Nagpur
