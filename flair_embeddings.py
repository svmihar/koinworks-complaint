from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings
from flair.data import Sentence

# initialize the word embeddings
glove_embedding = WordEmbeddings("id")
flair_embedding_forward = FlairEmbeddings('models/')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([glove_embedding, flair_embedding_forward])


def get_tweet_embeddings(tweet: str):
    tweet_s = Sentence(tweet)
    document_embeddings.embed(tweet_s)
    return sentence.embedding