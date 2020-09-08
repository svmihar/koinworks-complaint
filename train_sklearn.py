import joblib
from util import load_label, model_path, data_path, asset_path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from cleaning import s


@dataclass
class rf_model:
    vectorizer = joblib.load(
        asset_path / "tfidf_vectorizer.pkl"
    )  # vectorizer dumped on cleaning.py
    model = joblib.load(asset_path / "classifier.pkl")

    def process_tweet(self, tweet):
        return s(tweet)

    def predict(self, query):
        query_c = self.process_tweet(query)
        query_v = self.vectorizer.transform([query_c])
        return self.model.predict(query_v)


# classifier
def train_classifier():
    X_train, y_train, X_test, y_test = load_label(split=True)
    vectorizer = joblib.load(
        data_path / "tfidf_vectorizer.pkl"
    )  # vectorizer dumped on cleaning.py
    x = vectorizer.transform(X_train)
    classifier = RandomForestClassifier(max_depth=10, random_state=42)
    classifier.fit(x, y_train)
    joblib.dump(classifier, model_path / "classifier.pkl")
    test_array = vectorizer.transform([X_train[1]])
    test = classifier.predict(test_array)
    print(
        accuracy_score(
            [classifier.predict(vectorizer.transform([a]).toarray()) for a in X_test],
            y_test,
        )
    )
