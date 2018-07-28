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
import matplotlib
%matplotlib inline
%config InlineBackend.figure_format = 'svg' 
plt.style.use('bmh')
sns.set()  # defines the style of the plots to be seaborn style

df = pd.read_json('/home/asuerdem/Documents/ai_culture/CN_afterJaccard.json')




""" 3. Tokenize words and preprocess the text"""


"""3.1  Prepare Stopwords"""
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = list(STOP_WORDS) # <- set of Spacy's default stop words
stop_words.extend(['imgs', 'syndigate', 'info', 'jpg', 'http', 'photo','eca', 'nd', 'th', 'st', 
                   'system', 'time', 'year', 'people', 'world', 'technology', "telegraph", "quote_component", "emded_component", "float_left", "guardian_media",
                   "daily_mail", "datum", "min_width", 'post_publisher', 'nonascii', "margin_left", "margin_right", "html_embed",
                   "embed_component","network_content", "china_daily", "china_morniing", "copyright_south"])


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))

"""3.1. Creating Bigram and Trigram Models
3.1.1 Build the bigram and trigram models gives a warning, control what it is and solve"""
bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
"""See trigram example"""
#print(trigram_mod[bigram_mod[data_words[9]]])

"""3.1.2 . Remove Stopwords, Make Bigrams and Lemmatize
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
         
"""Remove more Stop Words, find ways to remove a long list of SWS, not one, this only adds one sw"""
STOP_WORDS.add("artificial_intelligence")
data_lemmatized = remove_stopwords(data_lemmas)# Rem


"""4. Create the Dictionary and Corpus and BOW
# 4.1 Create Dictionary"""
id2word = corpora.Dictionary(data_lemmatized) 
"""size of dictionary"""
print("Found {} words.".format(len(id2word.values())))
""" corpus"""  
corpus = [id2word.doc2bow(text) for text in data_lemmatized]
"""check sparsity, for instance, sparse = .99  tokens which are missing from more than 99 of the documents in the corpus. """
data_dense = gensim.matutils.corpus2dense(corpus, num_docs=len(corpus),
                                  num_terms= len(id2word.values()))
print("Sparsicity: ", 1.0 - (data_dense > 0).sum() / data_dense.size)
"""
#sklearn vectorizer
lemmas = data_lemmas.apply(lambda x: ' '.join(x))
vect = sklearn.feature_extraction.text.CountVectorizer()
features = vect.fit_transform(lemmas)
data_dense = features.todense()
print("Sparsicity: ", 1.0 - (data_dense > 0).sum() / data_dense.size)
"""

""" 4.2 to make the corpus denser,  filter extremes, before this need to intrıduce zipfs law abuot configuration"""
id2word.filter_extremes(no_above=0.95, no_below=10)
id2word.compactify()  # Reindexes the remaining words after filtering
print("Left with {} words.".format(len(id2word.values())))
"""size of dictionary"""
print("Found {} words.".format(len(id2word.values())))
corpus = [id2word.doc2bow(text) for text in data_lemmatized]

"""check sparsity"""
data_dense = gensim.matutils.corpus2dense(corpus, num_docs=len(corpus),
                                  num_terms= len(id2word.values()) )
print("Sparsicity: ", 1.0 - (data_dense > 0).sum() / data_dense.size)


"""toturn the corpus into DF"""
corp = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:]]
    
""" 4.3 tfiidf, vectorizing"""        
from gensim.models import TfidfModel
tfidf_model = gensim.models.TfidfModel(corpus=corpus,
                                        id2word=id2word)
corpt = [[(id2word[id], freq) for id, freq in cp] for cp in  tfidf_model[corpus]]

"""ADD TF ALSO"""


"""4.4 to write the output into df"""
dat = [pd.DataFrame(el) for el in corp]
dat = [pd.DataFrame(fd) for fd in dat]
counter = 0
for fd in dat:
   fd.columns = ['word','count']
   fd['docid'] = [counter] * fd.shape[0]
   counter += 1
output = pd.concat(dat)
#merge the columns in dataframe
df['docid'] = df.index
output =pd.merge(df, output, on = "docid")
output = output.drop(columns ='content')


"""to filter the bigrams only"""
bigr = output[output['word'].str.contains("_")]

"""FROM THIS PART, 2 STRATEGIES, SAVE THE OUTPUT AND CONTINUE W R OR GO AHEAD W PYTHON"""




"""5 plotting"""
"""5 1 aggregating for plotting"""
from dplython import (DplyFrame, X, diamonds, select, sift, sample_n,
    sample_frac, head, arrange, mutate, group_by, summarize, DelayFunction) 
dfr = DplyFrame(output)
dfr = (dfr >> 
  group_by(X.word, X.source) >> 
  summarize(tot=X.count.sum()))
dff = (dfr >>select(X.word, X.tot ))

"""5.2 wordcloud"""
"""turns the word freq to dict"""
d = {}
for a, x in dff.values:
    d[a] = x
wordcloud = WordCloud(width = 1000, height = 1000,
                background_color ='white',
                min_font_size =15, max_font_size=120).generate_from_frequencies(frequencies=d)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

"""stacked bar plot"""
dfx = (dfr >>
arrange(-X.tot))
dfx=dfx.head(50)

from plotnine import *
(ggplot(dfx, aes(x='word', y='tot', fill='source'))
 + geom_col()  +
 theme(axis_text_x=element_text(rotation=45, hjust=1))
)
"""each newspaper"""
dfr = DplyFrame(output)
df_tele =(dfr >>sift(X.source=="guardian"))
df_tele = (df_tele >> 
  group_by(X.word, X.source) >> 
  summarize(tot=X.count.sum()))

df_tele = (df_tele >>select(X.word, X.tot ))
d = {}
for a, x in dff.values:
    d[a] = x
 
wordcloud = WordCloud(width = 1000, height = 1000,
                background_color ='white',
                min_font_size =10, max_font_size=150).generate_from_frequencies(frequencies=d)
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()





 
   





    



