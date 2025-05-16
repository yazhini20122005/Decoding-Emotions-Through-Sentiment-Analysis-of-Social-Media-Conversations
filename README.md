# Decoding Emotions Through Sentiment Analysis of Social Media Conversations

## 📌 Project Overview

This project aims to analyze and classify human emotions from social media conversations using advanced Natural Language Processing (NLP) techniques. By decoding textual data from platforms like Twitter, Reddit, and Facebook, the system identifies six primary emotions—**happiness, sadness, anger, fear, surprise, and disgust**—and visualizes emotional trends over time and across demographics.

## 👩‍💻 Team Members

- **Yalini R** – Data Cleaning, Documentation, Codebase Development  
- **Iswariya C** – Exploratory Data Analysis, Visualization, Code Implementation  
- **Sanjay M** – Feature Engineering, Reporting, Debugging Support  

## 🎯 Objectives

### Primary:
- Build a robust multi-class sentiment classifier for emotion detection.
- Achieve ≥ 85% accuracy.
- Develop a dashboard to track emotional trends across platforms.

### Secondary:
- Analyze emotional responses to events.
- Study emotion-based linguistic patterns.
- Detect sarcasm and mixed emotions.

## 🧠 Methodology

### 🔁 Workflow
1. **Data Collection** – APIs from Twitter/X, Reddit, Facebook, public datasets.
2. **Preprocessing** – Tokenization, emoji handling, hashtag segmentation, sarcasm detection.
3. **EDA** – Frequency, polarity, network, and trend analysis.
4. **Feature Engineering** – BoW, TF-IDF, word embeddings, emoji embeddings.
5. **Modeling** – Traditional ML (Naive Bayes, SVM, Random Forest) and Deep Learning (LSTM, CNN, BERT).
6. **Evaluation** – Confusion matrix, ROC curves, SHAP/LIME analysis.
7. **Deployment** – Flask API, Dashboard with Plotly, Dash, and Docker.
8. **Monitoring** – Real-time feedback, performance tracking.

## 🧾 Dataset

- ~1 million posts (2022–2024)
- Sources: Twitter, Reddit, Facebook, SemEval, ISEAR, TEC
- Languages: English (primary), Spanish, French
- Annotated using manual and semi-supervised techniques

## 🧹 Preprocessing Highlights

- Hashtag decomposition
- Emoji → text mapping
- Sarcasm flagging
- Spelling normalization
- Context linking for conversation threads

## 📊 Exploratory Data Analysis

- Word clouds by emotion
- Heatmaps and time series plots
- Platform-wise and demographic comparisons
- Topic-emotion mapping
- Emotion response to global events

## 🧰 Tools & Technologies

### Languages:
- Python

### Libraries & Frameworks:
- Pandas, NumPy, NLTK, SpaCy, Scikit-learn, TensorFlow, PyTorch, Hugging Face, XGBoost, LightGBM

### Databases:
- PostgreSQL, MongoDB, AWS S3

### Visualization:
- Seaborn, Plotly, Dash, Tableau

### Deployment:
- Flask, Docker, Kubernetes, AWS

### DevOps:
- GitHub Actions, MLflow, DVC, Weights & Biases

## 📈 Performance & Evaluation

- Cross-validation and transfer learning used
- Balanced classes with weighted loss
- Model performance tracked using learning curves, SHAP, LIME, and confusion matrices
- Dashboard includes real-time emotion tracking and drill-down analysis

## 🔒 Ethical Considerations

- Compliance with platform policies
- Anonymized user data
- Exclusion of private conversations
- Regular bias audits

## 🌐 GitHub Repository

[GitHub Repository Link](https://github.com/yazhini20122005/Decoding-Emotions-Through-Sentiment-Analysis-of-Social-Media-Conversations)

---

> © 2025 Yalini R – Sri Ramanujar Engineering College, Department of AI & DS
