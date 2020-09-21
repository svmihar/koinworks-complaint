import os
import numpy as np
from collections import Counter
from pprint import pprint

from tqdm.auto import tqdm
from util import data_path
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

"""
clustering topics
"""
EMBEDDING_TYPES = ["pca", "umap"]


def get_top_k_word(tweets: list, k: int = 10):
    words = [b for a in tweets for b in a.split()]
    return [a[0] for a in Counter(words).most_common(k)]


def check_column(column_name, df):
    return column_name in df.columns


def kmeans_(df):
    for emb in tqdm(EMBEDDING_TYPES, desc="kmeans"):
        X = np.array([a for a in df[emb].values])
        range_n_cluster = [x for x in range(4, 20, 2)]
        clusters = [KMeans(n_clusters=n) for n in range_n_cluster]
        labels = [cluster.fit_predict(X) for cluster in clusters]
        scores = [silhouette_score(X, label) for label in labels]
        best_by_index = np.argmax(scores)
        print(
            f"kmeans' highest silhouette score is {scores[best_by_index]}\n \
                with n_cluster: {range_n_cluster[best_by_index]} "
        )
        df[f"kmeans_{emb}"] = labels[best_by_index]
    return df


def dbscan_(df):
    for emb in tqdm(EMBEDDING_TYPES, desc="dbscan"):
        X = np.array([a for a in df[emb].values])
        db = DBSCAN(eps=0.005, min_samples=3)  # 3-> 2 * n - 1
        db.fit(X)
        labels = db.labels_
        df[f"dbscan_{emb}"] = labels
        print(f"dbscan n_cluster {len(set(labels))}")
    return df


if __name__ == "__main__":
    df = pd.read_pickle(data_path / "1_koinworks_vectorized.pkl")
    tes = df.pipe(kmeans_).pipe(dbscan_)
    df.to_pickle(data_path / "2_koinworks_clustered")
    # tes.to_pickle(data_path / "4_hasil_cluster.pkl")
