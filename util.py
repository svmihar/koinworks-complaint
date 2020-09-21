from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import os

data_path = Path("./data")
model_path = Path("./models/")
asset_path = Path("./asset/")
classifier_path = Path("./models/classifier")


def load_label(split=False):
    df = pd.read_csv(data_path / "eda.csv")
    df.dropna(inplace=True, subset=["complaint"])
    df = df[["cleaned", "complaint"]]
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(lambda x: "complaint" if x > 0 else "not_complaint")
    if split:
        train, test = train_test_split(df)
        X_train, y_train = train["text"].values, train["label"].values
        X_test, y_test = test["text"].values, test["label"].values
        return (X_train, y_train, X_test, y_test)
    return df


def make_dataset(csv="eda.csv"):
    x, y = train_test_split(load_label())
    y_test, y_valid = train_test_split(y)
    x.to_csv(data_path / "train.csv", index=False)
    y_test.to_csv(data_path / "test.csv", index=False)
    y_valid.to_csv(data_path / "dev.csv", index=False)


if __name__ == "__main__":
    os.system("rm -rf /tmp/*")
    os.system("rm -rf /root/.flair/embeddings/")
    os.system("rm -rf /root/ktrain_data")
