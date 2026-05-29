# Emotion Analysis & Text Clustering using Machine Learning

## Overview

This project focuses on Natural Language Processing (NLP) and Machine Learning using the GoEmotions Dataset.

Two different machine learning approaches were implemented:

1. Supervised Multi-Label Emotion Classification
2. Unsupervised Text Clustering using K-Means

The project uses TF-IDF vectorization for text representation and UMAP dimensionality reduction for visualization.

---

# Dataset

Dataset used: GoEmotions Dataset

The dataset contains text samples labeled with different emotions.

## Selected Emotion Labels

The supervised model uses the following emotions:

- anger
- confusion
- disappointment
- disapproval
- excitement
- love
- sadness

---

# Technologies & Libraries

## Python Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- catboost
- umap-learn

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn catboost umap-learn
```

---

# Project 1 — Multi-Label Emotion Classification

## Objective

The goal of this model is to predict multiple emotions from a single text input.

Example:

```text
"I miss you so much and I feel sad."
```

Possible output:

```text
love + sadness
```

Since one sentence can contain multiple emotions, this is treated as a:

### Multi-Label Classification Problem

---

# Workflow

## 1. Data Cleaning

The dataset is cleaned by:

- Removing missing text values
- Removing empty rows
- Removing duplicates
- Keeping only rows with at least one emotion label

---

## 2. Train/Test Split

The dataset is divided into:

- 90% training data
- 10% testing data

Using:

```python
train_test_split()
```

---

## 3. TF-IDF Vectorization

Text is converted into numerical features using:

```python
TfidfVectorizer
```

Settings used:

```python
max_features=10000
stop_words='english'
ngram_range=(1,2)
```

This allows the model to understand:

- Single words
- Word combinations (bigrams)

---

## 4. UMAP Dimensionality Reduction

UMAP is used to reduce the TF-IDF vectors into 2D space for visualization.

Purpose:

- Visualize text similarity
- Observe emotion distribution
- Create scatter plots

---

# Machine Learning Models Used

## 1. Logistic Regression

Implemented using:

```python
OneVsRestClassifier(LogisticRegression())
```

### Features

- Fast training
- Strong baseline performance
- Good for sparse TF-IDF data

### Evaluation Metrics

- F1 Score (Micro)
- F1 Score (Macro)
- Hamming Loss

---

## 2. CatBoost Classifier

Implemented using:

```python
OneVsRestClassifier(CatBoostClassifier())
```

### Features

- Gradient boosting algorithm
- Handles complex patterns
- Strong classification performance

### Parameters Used

```python
iterations=300
learning_rate=0.1
depth=6
```

---

## 3. Decision Tree Classifier

Implemented using:

```python
OneVsRestClassifier(DecisionTreeClassifier())
```

### Features

- Easy to interpret
- Tree-based classification
- Handles non-linear relationships

---

# Visualizations

## UMAP Scatter Plots

The project visualizes emotion distributions using:

- Logistic Regression scatter plot
- CatBoost scatter plot
- Decision Tree scatter plot

Each point represents a text sample projected into 2D space.

Color intensity represents:

```text
Number of emotions assigned to the text
```

---

## Emotion Distribution Bar Charts

Bar plots display:

- Predicted emotion counts
- Emotion frequency distribution

This helps analyze:

- Model prediction behavior
- Emotion imbalance

---

# Project 2 — Unsupervised Text Clustering

## Objective

This model groups similar text samples together without using labels.

Algorithm used:

```text
K-Means Clustering
```

---

# Workflow

## 1. Text Cleaning

The text is:

- Converted to lowercase
- Stripped of whitespace
- Cleaned from empty values
- Deduplicated

---

## 2. TF-IDF Vectorization

The text is converted into numerical vectors using:

```python
TfidfVectorizer()
```

Parameters:

```python
max_df=0.9
min_df=5
ngram_range=(1,2)
```

---

## 3. K-Means Clustering

K-Means divides the text data into clusters.

### Number of Clusters

```python
k = 8
```

Each cluster groups similar sentences together based on textual similarity.

---

## 4. UMAP Visualization

UMAP reduces the TF-IDF vectors into 2D space for visualization.

The scatter plot shows:

- Cluster separation
- Text grouping patterns
- Cluster density

---

## 5. Cluster Distribution Plot

A bar chart displays:

- Number of samples in each cluster

This helps evaluate cluster balance.

---

# Elbow Method

The code also includes an optional implementation of the:

## Elbow Method

Purpose:

- Find the optimal number of clusters (K)

The method calculates:

```text
Inertia
```

for different K values.

The optimal K is typically where the curve begins to flatten.

---

# Machine Learning Concepts Used

This project demonstrates:

- Natural Language Processing (NLP)
- Multi-label classification
- Unsupervised learning
- Text vectorization
- TF-IDF
- Dimensionality reduction
- Clustering
- Data visualization

---

# Possible Improvements

Future improvements may include:

- Deep Learning models (LSTM, BERT, Transformers)
- Hyperparameter tuning
- More emotion labels
- Better clustering evaluation
- Interactive dashboards
- Model saving/loading
- Real-time prediction API

---

# How to Run

## Step 1 — Install Libraries

```bash
pip install pandas numpy matplotlib scikit-learn catboost umap-learn
```

---

## Step 2 — Update Dataset Path

Replace:

```python
C:\...\go_emotions_dataset.csv
```

with your dataset location.

---

## Step 3 — Run the Scripts

```bash
python filename.py
```

---

# Output Examples

The project generates:

- F1 scores
- Hamming loss
- UMAP scatter plots
- Emotion prediction charts
- K-Means cluster visualization
- Cluster distribution graphs

---

# Author

Created by Tarek Gharzeddine  
Machine Learning & NLP Project
