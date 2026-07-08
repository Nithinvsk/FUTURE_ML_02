# 🎫 Support Ticket Classification & Priority Prediction System

### Future Interns – Machine Learning Task 2 (2026)

**Status:** ✅ Completed

**Objective:**
Develop a Natural Language Processing (NLP) pipeline that automatically analyzes customer support tickets and predicts both:

* **Ticket Category** (Billing Inquiry, Technical Issue, Refund Request, etc.)
* **Ticket Priority** (Critical, High, Medium, Low)

The system automates support ticket triaging, reducing manual effort and enabling faster customer response times.

---

# 🚀 Project Overview

Customer support organizations receive a large volume of customer requests every day. Manual ticket routing is time-consuming, inconsistent, and delays responses to critical issues.

This project implements a complete machine learning workflow capable of learning from historical support tickets and automatically predicting:

* Ticket Category
* Ticket Priority

The project follows an end-to-end NLP pipeline including data preprocessing, feature extraction, model comparison, evaluation, model persistence, and prediction.

---

# 🏗 Engineering Approach

Instead of relying on a single classifier, this project evaluates multiple machine learning algorithms to determine the best-performing model for each prediction task.

The workflow consists of:

1. Data Loading
2. Text Cleaning
3. Feature Engineering
4. Model Training
5. Model Comparison
6. Performance Evaluation
7. Model Serialization
8. Real-Time Ticket Prediction

This modular architecture makes the project easy to maintain, extend, and deploy.

---

# 📊 Dataset Analysis

Dataset Used:

**Customer Support Ticket Dataset (Kaggle)**

The dataset contains structured customer support tickets including:

* Ticket Subject
* Ticket Description
* Ticket Type
* Ticket Priority
* Ticket Status
* Resolution
* Ticket Channel

For this project, the **Ticket Subject** and **Ticket Description** are combined into a single text feature before preprocessing.

---

# 🧹 Data Preprocessing Pipeline

The text preprocessing pipeline performs several cleaning operations before feature extraction.

### Text Cleaning

* Convert text to lowercase
* Remove punctuation
* Remove numerical values
* Remove stop words
* Normalize whitespace

### Text Combination

```
Ticket Subject
        +
Ticket Description
        ↓
Combined Text
```

This improves the amount of contextual information available to the classifier.

---

# 📐 Feature Engineering

The cleaned text is transformed into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF measures the importance of a word in a document relative to the entire dataset.

Advantages:

* Reduces influence of common words
* Highlights discriminative terms
* Efficient sparse representation
* Suitable for classical NLP classifiers

The vectorizer is configured with:

* Maximum Features: **5000**
* Word-based tokenization

---

# 🤖 Machine Learning Models

Four supervised learning algorithms were trained and compared.

### 1. Logistic Regression

Used as a strong baseline linear classifier for sparse text features.

---

### 2. Multinomial Naive Bayes

A probabilistic classifier commonly used for document classification.

---

### 3. Linear Support Vector Machine (Linear SVM)

Optimizes a maximum-margin decision boundary for high-dimensional feature spaces.

---

### 4. Random Forest Classifier

An ensemble learning algorithm using multiple decision trees.

---

The model with the highest accuracy is automatically selected and stored for future inference.

---

# 📈 Evaluation Methodology

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

Confusion matrices are automatically generated and stored for visual analysis.

---

# 💾 Model Persistence

After training, the following artifacts are generated:

```
models/
│
├── category_model.pkl
├── priority_model.pkl
└── tfidf.pkl
```

The trained models are serialized using **Joblib**, allowing future predictions without retraining.

---

# 🔄 Prediction Pipeline

```
Customer Ticket
        │
        ▼
Ticket Subject
        +
Ticket Description
        │
        ▼
Text Cleaning
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Category Classifier
        │
        ├────────► Ticket Category
        │
        ▼
Priority Classifier
        │
        └────────► Ticket Priority
```

---

# 📂 Repository Structure

```
Support-Ticket-Classification/
│
├── dataset/
│   └── customer_support_tickets.csv
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── tfidf.pkl
│
├── outputs/
│   ├── category_confusion_matrix.png
│   ├── priority_confusion_matrix.png
│   ├── category_report.txt
│   └── priority_report.txt
│
├── src/
│   ├── preprocess.py
│   ├── train_category.py
│   ├── train_priority.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Nithinvsk/FUTURE_ML_02.git
```

Move into the project

```bash
cd Support-Ticket-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train Models

```bash
python main.py
```

This will:

* Load the dataset
* Clean the text
* Generate TF-IDF features
* Train multiple classifiers
* Compare model performance
* Save trained models
* Generate evaluation reports

---

### Predict New Tickets

```bash
python src/predict.py
```

Example

**Input**

Subject

```
Unable to Login
```

Description

```
I cannot log in to my account after resetting my password.
```

**Output**

```
Category : Technical issue
Priority : Low
```

---

# 📊 Project Outputs

The project automatically generates:

```
outputs/

category_confusion_matrix.png

priority_confusion_matrix.png

category_report.txt

priority_report.txt
```

These reports provide a detailed view of classifier performance.

---

# 🔮 Future Enhancements

* Hyperparameter Optimization using GridSearchCV
* Word Embeddings (Word2Vec / FastText)
* Transformer-based Models (BERT)
* Streamlit Dashboard
* REST API using FastAPI
* Docker Containerization
* Cloud Deployment
* Real-Time Ticket Monitoring

---

# 💼 Skills Demonstrated

* Natural Language Processing (NLP)
* Text Preprocessing
* TF-IDF Feature Engineering
* Machine Learning Classification
* Multi-Model Evaluation
* Model Serialization
* Python Programming
* Modular Software Architecture
* Data Visualization
* Software Engineering Best Practices

---

# 👨‍💻 Author

**Nithin**

Machine Learning | NLP | Python | Scikit-learn

Developed as part of **Future Interns – Machine Learning Task 2 (2026).
