import os
import pandas as pd
import nltk
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from nltk.util import bigrams
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora, models


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('wordnet_ic')

from nltk.collocations import *


def preprocess(document, include_bigrams=False):
    stop_words_english = set(stopwords.words('english'))
    stop_words_german = set(stopwords.words('german'))
    combined_stop_words = stop_words_english.union(stop_words_german)
    tokens = word_tokenize(document.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in combined_stop_words]
    if include_bigrams:
        bigram_tokens = ['_'.join(bg) for bg in bigrams(tokens)]
        return tokens + bigram_tokens
    else:
        return tokens


def top_bigrams(corpus):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(corpus)
    finder.apply_freq_filter(2)
    top_bigrams = finder.nbest(bigram_measures.pmi, 50)
    return top_bigrams
  

def lda_topic(corpus, num_topics=20, num_words=10):
    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(doc) for doc in corpus]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=50)
    return (ldamodel.print_topics(num_topics=num_topics, num_words=num_words)) #num_words give me top words from this topic

if __name__ == '__main__':
    
    df = pd.read_csv('ausland_reviews.csv')

    #Create a corpus of tokens (a list of unigrams/bigrams)
    corpus = []
    for text in df['text_review']:
        processed_tokens = preprocess(text)
        corpus.extend(processed_tokens)
    print('Corpus ', len(corpus), corpus[0:50])

    # Prepare a corpus for Word2Vec and LDA (list of tokenized documents) 
    unigram_doc_corpus = df['text_review'].apply(preprocess).tolist() #create corpus for LDA topic modelling
    bigram_doc_corpus = df['text_review'].apply(lambda x: preprocess(x, include_bigrams=True)).tolist()

    print(f"Top bigrams for ", top_bigrams(corpus)) #finds most popular bigrams in text
    print(f"Topic modeling for  ", lda_topic(bigram_doc_corpus, num_topics=5, num_words=10)) #creats topics, with more relevant words for this topic








