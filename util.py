from pathlib import Path
import os

data_path = Path("./data")
model_path = Path("./models/")
classifier_path = Path("./models/classifier")
flair_datapath = data_path / "flair_format"
train_flair_datapath = flair_datapath / "train"
if not train_flair_datapath.is_dir():
    train_flair_datapath.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    os.system("rm -rf /tmp/*")
    os.system("rm -rf /root/.flair/embeddings/")
    os.system("rm -rf /root/ktrain_data")