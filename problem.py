#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:22:30 2018

@author: asuerdem
"""



# more common imports
import pandas as pd
import numpy as np
from collections import Counter
from pprint import pprint
import re

# languange processing imports
import nltk
from gensim.corpora import Dictionary
# preprocessing imports
from sklearn.preprocessing import LabelEncoder

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec


# spacy for lemmatization
import spacy
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# visualization imports
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
%matplotlib inline
%config InlineBackend.figure_format = 'svg' 
plt.style.use('bmh')
sns.set()  # defines the style of the plots to be seaborn style

df = pd.read_json('/home/asuerdem/Documents/ai_culture/UK_afterJaccard.json')


"""remove non-ascii words"""
df = df.reset_index(drop=True)
our_special_word = 'nonascii'
def remove_ascii_words(df):
  non_ascii_words = []
  for i in range(len(df)):
        for word in df.loc [i, 'content'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc [i, 'content'] = df.loc[i, 'content'].replace(word, our_special_word)
  return non_ascii_words

non_ascii_words = remove_ascii_words(df)
print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(
    len(non_ascii_words)))

"""spell corrector"""
from textblob import TextBlob
df['content'][:5].apply(lambda x: str(TextBlob(x).correct()))
"""wordcloud to see extreme words, if nonsense"""

all_text = ' '.join([text for text in df['content']])
print('Number of words in all_text:', len(all_text))
from wordcloud import WordCloud
wordcloud = WordCloud( width=800, height=500,
                      random_state=21, max_font_size=110).generate(all_text)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


"""CLEANER, ADD more regexes, more soecific to news
Convert content to list, and clean the text with regex"""
data = df.content.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', content) for content in data]
# Remove new line characters
data = [re.sub('\s+', ' ', content) for content in data]
data = [re.sub('\n+', ' ', content) for content in data]
data = [re.sub('\r+', ' ', content) for content in data]
# Remove distracting single quotes
data = [re.sub("\'", "", content) for content in data]

#pprint(data[:1])


"""5. Prepare Stopwords"""
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = list(STOP_WORDS) # <- set of Spacy's default stop words
stop_words.extend(['imgs', 'syndigate', 'info', 'jpg', 'http', 'photo','eca', 'nd', 'th', 'st', 
                   'system', 'time', 'year', 'people', 'world', 'technology', 
                   'post_publisher', 'nonascii'])
""" Tokenize words and preprocess the text"""

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))


"""9. Creating Bigram and Trigram Models
Build the bigram and trigram models"""
bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
"""See trigram example"""
#print(trigram_mod[bigram_mod[data_words[9]]])



"""10. Remove Stopwords, Make Bigrams and Lemmatize
Define functions for stopwords, bigrams, trigrams and lemmatization"""
nlp = spacy.load("en")
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV', 'VERB']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return pd.Series(texts_out)

"""Remove Stop Words"""
data_words_nostops = remove_stopwords(data_words)# Remove Stop Words
""" Form Bigrams"""
data_words_bigrams = make_bigrams(data_words_nostops)

""" Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en"""
nlp = spacy.load('en', disable=['parser', 'ner'])
"""Do lemmatization keeping only noun, adj, vb, adv"""
data_lemmas = lemmatization(data_words_bigrams, allowed_postags=['NOUN' ]) #, 'VERB', 'ADV', 'ADJ',

"""PROBLEM: BUNUN ÇIKTISININ DATA_LEMMATIZED İLE AYNI FORMATTA OLMASINI ISTIYORUM. YANİ, 
DATA_LEMMATIZED DAN TEK KELİME OLANLAR FILTER OUT OLACAK, ŞUNU YAPTIM AMA TEK BİR LŞSTE ÇIKARIYOR. BEN DÖKÜMANLAR KELİME LİSTSİ
TÜM CORPUS DA DOCS LİTESİ OLSUN İSTİYORUM, DATA_LEMMAS GİBİ" 


for idx in range(len(data_lemmas)):
    for token in bigram[data_lemmas[idx]]:
        if '_' in token:
          print(token)
          

