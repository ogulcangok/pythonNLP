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


"""1) FEATURE INSPECTION 
This exploratory phase is always a good thing to start with to get an idea of the data and 
it will probably help you out making decisions later on in the process of 
any data science problem you have and in particular now for text classification.
"""
""" 1.1 check if there's missing data, bu dedupANDclean konabilir"""
df.isnull().sum()
"""sil diye bir şey ekenilir, na omit)

""" 1.2 Inspect source variable, ayrıca diğer categoric variables için de yapılabilir"""
"""check what sources we have, burada eğer çok az sayıda olan source varsa çıkarma fonksiyonu eklenebilir"""
nmb = len(df.source.value_counts().index)

"""barchart of sources"""
fig, ax = plt.subplots(1,1,figsize=(8,6))
source_vc = df.source.value_counts()

ax.bar(range(nmb), source_vc)
ax.set_xticks(range(nmb))
ax.set_xticklabels(source_vc.index, fontsize=11)
for rect, c, value in zip(ax.patches, ['b', 'r'], source_vc.values): # , 'g', 'y', 'c', 'm'
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.5*height, value, ha='center', va='center', fontsize=10, color='white')

"""1.3 Inspect text variables (i.e title, content), min max number"""

"""1.3.1 özet"""
document_lengths = np.array(list(map(len, df.content.str.split(' '))))
print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))    

""" 1.3.2  word distribution by doc"""
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax);
"""detecting very short docs"""
print("There are {} documents with over 100 words.".format(sum(document_lengths > 100)))

"""1.3.3 distribution in shorter docs, this is to decide if any docs should be excluded"""
shorter_documents = document_lengths[document_lengths <= 100]
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(shorter_documents, bins=50, ax=ax);

""" 1.3.4 list of shorter docs, 50 in this case, but change according to the analyses above"""
df[document_lengths <= 50]
""" filter out  shorter docs)"""
df = df[document_lengths > 50]
    

"""MORE TO ADD, lexical diversity, readability, lexical richness etc..."""

"""2) Feature creation

The usual approach in natural language processing, is to first kind of cleanse the text. We have to make sure our model can understand similarities and understand when two different words mean similar things. You can't just input the raw text into a model and expect it to understand everything. Therefor, I summarized the most common steps to take in order to achieve this cleansing.
    Remove words that mean little; these are usually words that occur very frequently or words that occur very infrequently. 
    Also, punctuation can be removed, but perhaps you'd like to keep some punctuation in, 
    like exclamation marks or question marks since maybe one writer uses them more than others.
    Stemming; this basically means joining words that mean the same. Take for example the words running and runs, 
    the stem of both words is run. Thus with stemming you'd group these words together and give them the same meaning for the model.
    Vectorize words; since we can't input plain words into a model and expect it to learn from it, we have to vectorize the words. 
    This basically means create unit vectors for all words.

For LDA you'd like to perform all four steps. However, for w2v you'd only want to tokenize and remove some punctuation. 
The w2v model can determine by itself what words are important and what are not, but we'll get back to that later on."""

"""2.1 clean the text"""
"""2.1.1 spell corrector", this takes long time to finish, find if there is a faster way"""
from textblob import TextBlob
df['content'][:5].apply(lambda x: str(TextBlob(x).correct()))



""" 2.1.2 THIS WC IS EXPLORATORY ONLY,  
 to detect extreme words, irregular characters etc... if nonsense, then include them in the SW list"""
"""this is for flattening the list and bringing all strings into one text"""
all_text = ' '.join([text for text in df['content']])
print('Number of words in all_text:', len(all_text))
"""wordcloud"""
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = STOPWORDS,
                min_font_size = 20).generate(all_text)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

""" 2.1.3 remove non-ascii words"""
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


""" 2.1.4 CLEANER, ADD more regexes, more soecific to news"""
"""Convert content to list, and clean the text with regex"""
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



"""2.1.5. Prepare Stopwords"""
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = list(STOP_WORDS) # <- set of Spacy's default stop words
stop_words.extend(['imgs', 'syndigate', 'info', 'jpg', 'http', 'photo','eca', 'nd', 'th', 'st', 
                   'system', 'time', 'year', 'people', 'world', 'technology', "telegraph", "quote_component", "emded_component", "float_left", "guardian_media",
                   "daily_mail", "datum", "min_width", 'post_publisher', 'nonascii', "margin_left", "margin_right", "html_embed",
                   "embed_component","network_content", "china_daily", "china_morniing", "copyright_south"])

""" 3. Tokenize words and preprocess the text"""

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





 
   





    



