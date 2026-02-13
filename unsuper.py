import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\tareq\OneDrive\Desktop\go_emotions_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# -----------------------------
# 2. Keep ONLY text column + limit size
# -----------------------------
df = df[["text"]].head(50000)

# -----------------------------
# 3. Clean text data
# -----------------------------
df = df.dropna(subset=["text"])

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Remove empty strings
df = df[df["text"] != ""]

# Remove duplicate texts
df = df.drop_duplicates(subset="text")

# Reset index to keep alignment
df = df.reset_index(drop=True)

texts = df["text"]

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

# -----------------------------
# 5. K-Means clustering
# -----------------------------
k = 8
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# -----------------------------
# 6. 2D UMAP Mapping
# -----------------------------
umap_2d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42
)

X_umap_2d = umap_2d.fit_transform(X)

# -----------------------------
# 7. Scatter Plot
# -----------------------------
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_umap_2d[:, 0],
    X_umap_2d[:, 1],
    c=df["cluster"],
    s=5,
    alpha=0.6
)

plt.title("UMAP Projection of K-Means Clusters (TF-IDF)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(scatter, label="Cluster")
plt.show()

# -----------------------------
# 8. Bar Plot: Cluster Distribution
# -----------------------------
cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Samples")
plt.title("Cluster Distribution (K-Means)")
plt.xticks(cluster_counts.index)
plt.show()

'''
# Try K from 2 to 15
k_values = range(2, 16)
inertia = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X)  # X = TF-IDF matrix

    inertia.append(kmeans.inertia_)

# =============================
# Elbow Plot
# =============================
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.show()
'''