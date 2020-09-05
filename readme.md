# Koinworks compalints
this is basically a classifier turning out text generation

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
  - [eda.py](./eda.py)
    - implemented:
      - tfidf
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
flair model can be downloaded [here](https://drive.google.com/drive/u/5/folders/1uLGvvNCNAjAeOBPKyMwtLfEErBAsYuMQ)
### fasttext-id
- `WordEmbeddings('id-crawl')`

## search engine
### ~~annoy~~
- ~~make tree~~
exhaustive aja

## classifier
- train a tf model / fastai model
- onnx  / fastinference

### milvus
~~[milvus](https://milvus.io/)~~
- make embedding:
	- [x] tfidf, done `tfidf.pkl`
	- [x] fasttext
	- [x] flairembeddings
		- ~~ValueError: Found array with dim 3. check_pairwise_arrays expected <= 2. gak tau padahal gak adayang bikin dimensi 3~~ ganti ke scipy
- gak jadi pake milvus, soalnya dia ternyata framework yang jadi satu sama rest api nya

id nya ikut di `0_koinworks_raw.csv` udah dibikin `uuid4` biar gampang bikin indexernya

## potential complaint topics
### lda
1, 29, 7
- topics covers a range of complaints
	- cs not replying
	- website error
	- app error
	- **dana gak bisa ditarik**
	- **tiba tiba tenor berubah**

### kmeans
determining k, by using converged silhouette score, `check_topics_clustering.ipynb` on kmeans
dumb random shit. decided not to use it as a clustering method
- tested both in tfidf, and flair embeddings

### dbscan
9, sucks.

### top2vec
see /experiments/

## blog post ideas
- [ini buat opening](https://twitter.com/pakelagu/status/1292346337803923456)
	- meme: top: SHARE KODE KW
	- meme: bottom: KU TERTYPU OLEH KW
### extras
- aplikasinya sempet ilang juga lol  cek id: 517, 529, , cek tanggal, cek sumber
- dari search twitter sempet peak di 263 tweet di 04-02-2020 dan 09-01-2020
- dari kmeans, langsung kepisah dengan cantik 3 label

- siap, [didanai](https://money.kompas.com/read/2020/05/18/130309726/koinworks-dapat-pendanaan-rp-149-miliar-dari-perusahaan-inggris?utm_source=dlvr.it&utm_medium=twitter) tapi tenor kagak dibayar :)))
	- [sumber2](https://medium.com/lendable/koinworks-secures-us-10-million-from-lendable-to-support-indonesias-digital-smes-7119f42f7809)
	- [sumber3](https://internationalfinance.com/koinworks-secures-10-mn-funding-help-smes-raise-funds-online/)

- didanai lagi dong [woah](pic.twitter.com/ZbFjMJ3aSp)
