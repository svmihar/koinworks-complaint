from util import data_path, classifier_path
import pandas as pd
from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


def make_dataset(csv='eda.csv'): 
    from sklearn.model_selection import train_test_split 
    df = pd.read_csv(data_path/csv)
    df.dropna(inplace=True, subset=['complaint'])
    df = df[['cleaned', 'complaint']]
    df.columns=['text', 'label']
    x,y = train_test_split(df)
    y_test, y_valid = train_test_split(y)
    x.to_csv(data_path/'train.csv', index=False)
    y_test.to_csv(data_path/'test.csv', index=False)
    y_valid.to_csv(data_path/'dev.csv', index=False)

# check if training file is there
if 'train.csv' not in set(data_path.iterdir()): 
    classifier_path.mkdir(exist_ok=True)
    make_dataset()
    
# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label_topic"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(str(data_path),
                                         column_name_map,
                                         skip_header=True,
                                         delimiter=',',    
) 

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
tweet_embeddings = [FlairEmbeddings('models/best-lm.pt')]

# 4. initialize document embedding by passing list of word embeddings
document_embeddings = DocumentRNNEmbeddings(tweet_embeddings, 
                                            bidirectional = True, 
                                            rnn_type='lstm',
                                            rnn_layers=2,
                                            dropout=.25,
                                            hidden_size=256)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('models/classifier/',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)