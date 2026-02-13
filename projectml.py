# =============================
# 0. Imports
# =============================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, hamming_loss
import umap
import matplotlib.pyplot as plt

# =============================
# 1. Load Dataset and cleaning
# =============================
df = pd.read_csv(r"C:\Users\tareq\OneDrive\Desktop\go_emotions_dataset.csv")

# Limit to first 50,000 rows
df = df.head(50000)

text_col = "text"
emotion_cols = [
    "anger", "confusion",
    "disappointment", "disapproval",
    "excitement", "love",
    "sadness"
]

# Drop rows with missing text
df = df.dropna(subset=[text_col])

# Ensure text is string and strip whitespace
df[text_col] = df[text_col].astype(str).str.strip()

# Remove empty text rows
df = df[df[text_col] != ""]

# Remove duplicate texts
df = df.drop_duplicates(subset=text_col)

# Drop rows with missing emotion labels (safety)
df = df.dropna(subset=emotion_cols)

# Keep only rows with at least one emotion
df = df[df[emotion_cols].sum(axis=1) > 0]

df = df[[text_col] + emotion_cols]

# =============================
# 2. Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    df[text_col],
    df[emotion_cols],
    test_size=0.1,
    random_state=42
)

# =============================
# 3. TF-IDF Vectorization
# =============================
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =============================
# 4. Common UMAP Embedding for Visualization
# =============================
X_dense = X_train_tfidf.toarray()
umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.5,
    n_components=2,
    metric='cosine',
    random_state=42
)
X_umap = umap_model.fit_transform(X_dense)

# =============================
# 5a. Multi-label Logistic Regression
# =============================
lr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight='balanced')
)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

# Evaluation
print("\n--- Logistic Regression ---")
print("F1 (micro):", f1_score(y_test, y_pred_lr, average='micro'))
print("F1 (macro):", f1_score(y_test, y_pred_lr, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred_lr))

# Scatter plot
emotion_count_lr = y_train.sum(axis=1)
plt.figure(figsize=(10,7))
plt.scatter(
    X_umap[:,0],
    X_umap[:,1],
    c=emotion_count_lr,
    cmap='viridis',
    s=10,
    alpha=0.6
)
plt.colorbar(label="Number of emotions")
plt.title("UMAP Scatter - Logistic Regression (TF-IDF)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

# Bar plot
pred_counts_lr = y_pred_lr.sum(axis=0)
plt.figure(figsize=(8,5))
plt.bar(emotion_cols, pred_counts_lr, color='skyblue')
plt.title("Logistic Regression - Predicted Emotion Counts")
plt.ylabel("Number of Samples")
plt.show()

# =============================
# 5b. Multi-label CatBoost
# =============================

# CatBoost requires dense input
X_train_cb = X_train_tfidf.toarray()
X_test_cb  = X_test_tfidf.toarray()

cb_model = OneVsRestClassifier(
    CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='F1',
        verbose=0,
        random_seed=42
    )
)

cb_model.fit(X_train_cb, y_train)
y_pred_cb = cb_model.predict(X_test_cb)

# Evaluation
print("\n--- CatBoost ---")
print("F1 (micro):", f1_score(y_test, y_pred_cb, average='micro'))
print("F1 (macro):", f1_score(y_test, y_pred_cb, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred_cb))

# Scatter plot (same UMAP embedding)
cb_emotion_count = y_train.sum(axis=1)
plt.figure(figsize=(10,7))
plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=cb_emotion_count,
    cmap='plasma',
    s=10,
    alpha=0.6
)
plt.colorbar(label="Number of emotions")
plt.title("UMAP Scatter - CatBoost (TF-IDF)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

# Bar plot
pred_counts_cb = y_pred_cb.sum(axis=0)
plt.figure(figsize=(8,5))
plt.bar(emotion_cols, pred_counts_cb, color='purple')
plt.title("CatBoost - Predicted Emotion Counts")
plt.ylabel("Number of Samples")
plt.show()

# =============================
# 5c. Multi-label Decision Tree
# =============================
dt_model = OneVsRestClassifier(
    DecisionTreeClassifier(class_weight='balanced', random_state=42)
)
dt_model.fit(X_train_tfidf, y_train)
y_pred_dt = dt_model.predict(X_test_tfidf)

# Evaluation
print("\n--- Decision Tree ---")
print("F1 (micro):", f1_score(y_test, y_pred_dt, average='micro'))
print("F1 (macro):", f1_score(y_test, y_pred_dt, average='macro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred_dt))

# Scatter plot
dt_emotion_count = y_train.sum(axis=1)
plt.figure(figsize=(10,7))
plt.scatter(
    X_umap[:,0],
    X_umap[:,1],
    c=dt_emotion_count,
    cmap='coolwarm',
    s=10,
    alpha=0.6
)
plt.colorbar(label="Number of emotions")
plt.title("UMAP Scatter - Decision Tree (TF-IDF)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()

# Bar plot
pred_counts_dt = y_pred_dt.sum(axis=0)
plt.figure(figsize=(8,5))
plt.bar(emotion_cols, pred_counts_dt, color='green')
plt.title("Decision Tree - Predicted Emotion Counts")
plt.ylabel("Number of Samples")
plt.show()

