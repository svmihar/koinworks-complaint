# Koinworks compalints
this is basically a classifier turning out text generation

## datasource
[from twint raw](https://drive.google.com/file/d/1iUGPIYcZN1WA_sjut1um8AEdF_hniPlh/view?usp=sharing)
[with all embeddings](https://drive.google.com/drive/folders/1n61FktPgLxVb-GeZ_5l3xSFk_bcMqi7F?usp=sharing)


## real questions
1) What high-level trends can be inferred from Koinworks
tweets?
2) Are there any events that lead to spikes in koinworks
Twitter activity?
3) Which topics are distinct from each other?
disease outbreaks?

## approach
- scraping: twint
- preprocess:
  - [cleaning.py](./cleaning.py)
    - remove
      - [stopwords](./stopwords.txt)
      - punctuations
      - tweets from [koinworks](http://twitter.com/koinworks)
      - referral tweets (tweets promoting to use certain code to obtain voucher)
      - hyperlinks
    - then save it to `1_koinworks_cleaned.pkl`
    - implemented:
      - tfidf
      - custom flair embeddings
	  	- pretrained on: tweets itself
	  	- pooled with fasttext, flair embeddings
      - pca
      - umap
    - then save it to `2_koinworks_fix.pkl`
- `[tfidf, doc2vec, flair, fasttext]`
    - for flair embeddings, the tweets are "fine-tuned" using `pretrain.py` on `id-forward` [reference](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)
    - doc2vec using the `top2vec`
- [kmeans, hdbscan, lda2vec]
  - determine `n_topics` using "converged" silhouette score
- search keywords that represents a complaint
- clean the topics, so that all the tweets inside that topics are actually complaints
- train classifier
- train text generator

## EDA stuff
- harus ada `eda.csv` yang isinya:
  - date
  - complaint / not complaint
  - topic id per clustering method

## references
1. [Exploratory Analysis of Covid-19 Tweets using Topic Modeling, UMAP, and DiGraphs](https://arxiv.org/abs/2005.03082)


## pipeline
cleaning -> eda -> topic cluster -> labelling topics -> train models -> serve

## sources
### twitter search keywords
- [x] koinworks
- [x] koinwork
### news site?
- kompas?
- google?
dunno, will do if feeling cute lol
[1](https://swa.co.id/swa/trends/koinworks-catat-pertumbuhan-30-pasca-pelonggaran-psbb)

## EDA
- [x] top 100 most common unigram
- [x] top 100 most common bigram
- [x] top 100 most common trigram
- [x] wordcloud
- [x] maybe topic modelling with LDA
- [ ] distribusi kata yang merupakan keluhan
- [ ] topic trend in a given time
- visualize
	- ~~tfidf~~
	- ~~kmeans~~
	- flair (pca-ed lol)
	- ~~lda~~
	- ~~dbscan~~

## labelling
- search tweet with a definite "keluhan", then use cosine similarity to search similar ones, then label it too as keluhan
cek di `koinworks_labeled_lda.csv`

mostlikely keluhan keywords:
['telat', ]

## serving frontend
- classifier:
	- menentukan apakah tweet itu komplain atau nggak
		ada penjelasan: ini ada di modulenya ktrain
- dashboard:
	- daily keluhan berapa
	- top keywords keluhan
	- label buat graph nya, selain warna
- complaint generator
  - [ ] markov
  - [ ] lstm

----

## embeddings
this is a [pooled document embeddings](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/DOCUMENT_POOL_EMBEDDINGS.md) on:
### flair
- [x] pretrain with lm-forward + tweets
- [ ] make tweet encoder
flair model can be downloaded [here](https://drive.google.com/uc?id=1pb4JWy4ffUrDfnriAzePOrQmnmsYYPHE)
### fasttext-id
- `WordEmbeddings('id-crawl')`

## search engine
### ~~annoy~~
- ~~make tree~~
exhaustive aja

## classifier
- train a tf model / fastai model
- flair model: 
    - flair_embedding only, rnn (gru): 163M, 83%
    - flair_embeddings + id-forward: 
    ``` 
    By class:
              precision    recall  f1-score   support

         0.0     0.8312    0.7111    0.7665        90
         1.0     0.6176    0.7636    0.6829        55
   micro avg     0.7310    0.7310    0.7310       145
   macro avg     0.7244    0.7374    0.7247       145
    weighted avg     0.7502    0.7310    0.7348       145
     samples avg     0.7310    0.7310    0.7310       145
    ```
    
    - flair_embeddings only, rnn(lstm), hidden_layer=3, bidirectional: 
    ```
    Results:
    - F-score (micro) 0.7862
    - F-score (macro) 0.7791
    - Accuracy 0.7862

    By class:
                  precision    recall  f1-score   support

             1.0     0.6027    0.9565    0.7395        46
             0.0     0.9722    0.7071    0.8187        99

       micro avg     0.7862    0.7862    0.7862       145
       macro avg     0.7875    0.8318    0.7791       145
    weighted avg     0.8550    0.7862    0.7936       145
     samples avg     0.7862    0.7862    0.7862       145
     ```
 
     - flair_embeddings only, rnn(lstm), hidden_layer=3, bidrectional [model_link](https://drive.google.com/drive/u/0/folders/19ZG8jF8U9WnAY9gXXtQo42qgsl68XPxD): 
     ```
     Results:
    - F-score (micro) 0.8207
    - F-score (macro) 0.8197
    - Accuracy 0.8207

    By class:
                  precision    recall  f1-score   support

             0.0     0.9559    0.7386    0.8333        88
             1.0     0.7013    0.9474    0.8060        57

       micro avg     0.8207    0.8207    0.8207       145
       macro avg     0.8286    0.8430    0.8197       145
    weighted avg     0.8558    0.8207    0.8226       145
     samples avg     0.8207    0.8207    0.8207       145
     ```
     - document_embeddings = DocumentRNNEmbeddings(tweet_embeddings,  bidirectional = True,  rnn_type='lstm', rnn_layers=2, dropout=.25, hidden_size=256) 
     ```
     Results:
    - F-score (micro) 0.7862
    - F-score (macro) 0.7666
    - Accuracy 0.7862

    By class:
                  precision    recall  f1-score   support

             0.0     0.7879    0.8864    0.8342        88
             1.0     0.7826    0.6316    0.6990        57

       micro avg     0.7862    0.7862    0.7862       145
       macro avg     0.7852    0.7590    0.7666       145
    weighted avg     0.7858    0.7862    0.7811       145
     samples avg     0.7862    0.7862    0.7862       145
     
     ```
     - document_embeddings = DocumentRNNEmbeddings(tweet_embeddings,  bidirectional = True,  rnn_type='gru', rnn_layers=2, dropout=.25, hidden_size=256)[model_link](https://drive.google.com/drive/u/0/folders/1mVa-O8KoFqx1Y4X3VgzOlSqGs4eh5s8z)
     ```
     Results:
    - F-score (micro) 0.8414
    - F-score (macro) 0.8253
    - Accuracy 0.8414

    By class:
                  precision    recall  f1-score   support

             1.0     0.7647    0.7800    0.7723        50
             0.0     0.8830    0.8737    0.8783        95

       micro avg     0.8414    0.8414    0.8414       145
       macro avg     0.8238    0.8268    0.8253       145
    weighted avg     0.8422    0.8414    0.8417       145
     samples avg     0.8414    0.8414    0.8414       145
     
     ```
     
     
### extras
- aplikasinya sempet ilang juga lol  cek id: 517, 529, , cek tanggal, cek sumber
- dari search twitter sempet peak di 263 tweet di 04-02-2020 dan 09-01-2020
- dari kmeans, langsung kepisah dengan cantik 3 label

- siap, [didanai](https://money.kompas.com/read/2020/05/18/130309726/koinworks-dapat-pendanaan-rp-149-miliar-dari-perusahaan-inggris?utm_source=dlvr.it&utm_medium=twitter) tapi tenor kagak dibayar :)))
	- [sumber2](https://medium.com/lendable/koinworks-secures-us-10-million-from-lendable-to-support-indonesias-digital-smes-7119f42f7809)
	- [sumber3](https://internationalfinance.com/koinworks-secures-10-mn-funding-help-smes-raise-funds-online/)

- didanai lagi dong [woah](pic.twitter.com/ZbFjMJ3aSp)
