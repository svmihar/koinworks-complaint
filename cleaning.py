from sklearn.decomposition import PCA
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from top2vec import Top2Vec
import string
import re
from umap import UMAP
from util import data_path
import pandas as pd


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


def preprocess(df):
    print("cleaning started")

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

    df["cleaned"] = df["tweet"].apply(s)
    df["flair_dataset"] = df["tweet"].apply(s, remove_stopword=False)
    df["is_ref"] = df["cleaned"].apply(is_referral)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["is_ref"] == False]
    df = df.dropna(subset=["tweet", "username"])
    df = df.reset_index(drop=True)
    print("preprocess done")
    return df


def top2vec_vec(df):
    docs = df["cleaned"].values
    model = Top2Vec(docs, speed="deep-learn", workers=4)
    model.save("./models/top2vec.model")
    # topic_sizes, topic_nums = model.get_topic_sizes()
    print(
        f"vocab learned: {len(model.model.wv.vocab.keys())}"
    )  # kalo terlalu kecil word nya berarti harus diganti parameter sampling sama min_count nya di top2vec.py
    topic_id = None  # TODO: get topic id per tweet from top2vec
    tweet_vector = [a for a in model.model.docvecs.vectors_docs]
    breakpoint()
    df["doc2vec"] = tweet_vector
    df["top2vec_id"] = "TBA"
    return df


def reduce_dim(df):
    X = np.array([a for a in df["tfidf"].values])
    pca = PCA(n_components=2, svd_solver="full")
    # um = UMAP(n_components=5, n_neighbors=15, metric="euclidean")
    df["pca"] = [a for a in pca.fit_transform(X)]
    # df["umap"] = [a for a in um.fit_transform(X)]
    print('reducing dimension done')
    return df


def vectorize_tfidf(df):
    v = TfidfVectorizer()
    tfidf_vector = v.fit_transform(df.cleaned.values)
    df["tfidf"] = [x for x in tfidf_vector.toarray()]
    dump(v, "./data/tfidf_vectorizer.pkl")
    return df


def embedding_pipeline(df) -> pd.DataFrame:
    print("vectorizing")
    df.pipe(vectorize_tfidf).pipe(top2vec_vec)
    print("vectorizing done")
    return df


if __name__ == "__main__":
    df = load_data().pipe(preprocess).pipe(embedding_pipeline).pipe(reduce_dim)
    df.to_pickle(data_path / "1_koinworks_vectorized.pkl")
