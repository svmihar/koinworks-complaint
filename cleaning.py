from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
from util import data_path
import pandas as pd
from umap import UMAP


STOPWORDS = {a.replace("\n", "") for a in (open("stopwords.txt").readlines())}
UNALLOWED = list(string.digits) + list(string.punctuation)


def load_data():
    return pd.read_csv("./data/0_koinworks.csv")


def is_referral(tweet: str):
    p = False
    t = tweet.split()
    keywords = [
        "referral",
        "kode",
        "referal",
        "code",
        "click to watch",
        "youtube",
        "download",
        "gratis",
    ]
    for word in keywords:
        if word in t:
            return True
    return p


def s(tweet: str, remove_stopword: bool = True) -> str:
    # lowercase
    tweet_ = tweet.lower()
    # remove link
    tweet_ = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
        "",
        tweet_,
    )
    # remove digits
    for d in UNALLOWED:
        tweet_ = tweet_.replace(d, "")
    # remove multiple white space + split
    tweet_split = [t for t in tweet_.split() if t]
    if remove_stopword:
        return " ".join([t for t in tweet_split if t not in STOPWORDS])
    return " ".join([t for t in tweet_split])


def preprocess(df):
    print("cleaning started")

    df["cleaned"] = df["tweet"].apply(s)
    df["is_ref"] = df["cleaned"].apply(is_referral)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_ref"] == False]
    df = df.dropna(subset=["tweet", "username"])
    df = df.reset_index(drop=True)
    print("preprocess done")
    return df


def reduce(X):
    pca = PCA(n_components=2, svd_solver="full")
    um = UMAP(n_components=5, n_neighbors=15, metric="euclidean")
    X_pca = [a for a in pca.fit_transform(X)]
    X_umap = df["umap"] = [a for a in um.fit_transform(X)]
    return X_pca, X_umap


def reduce_dim(df):
    X = np.array([a for a in df["tfidf"].values])
    df["pca"], df["umap"] = reduce(X)
    print("reducing dimension done")
    return df


def vectorize_tfidf(df):
    v = TfidfVectorizer()
    tfidf_vector = v.fit_transform(df.cleaned.values)
    df["tfidf"] = [x for x in tfidf_vector.toarray()]
    dump(v, "./data/tfidf_vectorizer.pkl")
    return df


def embedding_pipeline(df) -> pd.DataFrame:
    print("vectorizing")
    df.pipe(vectorize_tfidf)  # .pipe(vectorize_flair)
    print("vectorizing done")
    return df


if __name__ == "__main__":
    df = load_data().pipe(preprocess).pipe(embedding_pipeline).pipe(reduce_dim)
    write_flair_dataset(df["flair_dataset"].values)
    df.to_pickle(data_path / "1_koinworks_vectorized.pkl")
