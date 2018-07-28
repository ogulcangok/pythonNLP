#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:09:25 2018

@author: asuerdem
"""

import pandas as pd
import re
import numpy as np
from collections import Counter
from pprint import pprint
#for plotting
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
sns.set() # defines the style of the plots to be seaborn style


dat = pd.read_json('/home/asuerdem/Documents/ai_culture/UK.json')

"""
FEATURE INSPECTION 
This exploratory phase is always a good thing to start with to get an idea of the data and 
it will probably help you out making decisions later on in the process of 
any data science problem you have and in particular now for text classification.
"""


""" 
1. check if there's missing data"
""
df.isnull().sum() #sil diye bir şey ekenilir, na omit

"""
2. Duplicates
"""

patterns = ['(.*)\s\(','(.*)\s-','(.*)\.co','(.*)\s-']
shingles = []
duplicates = []

def get_shingles(text, char_ngram=5):
    """Create a set of overlapping character n-grams.
    
    Only full length character n-grams are created, that is the first character
    n-gram is the first `char_ngram` characters from text, no padding is applied.

    Each n-gram is spaced exactly one character apart.

    Parameters
    ----------

    text: str
        The string from which the character n-grams are created.

    char_ngram: int (default 5)
        Length of each character n-gram.
    """
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))


def jaccard(set_a, set_b):
    """Jaccard similarity of two sets.
    
    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the two sets.

    Parameters
    ---------
    set_a: set
        Set of arbitrary objects.

    set_b: set
        Set of arbitrary objects.
    """
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


for news in dat['content']: shingles.append(get_shingles(news.lower()))
""" for each news in the dataframe's cloumn title, 
take it as lowercase make sets and push it to the shingles array. 
"""
for i_doc in range(len(shingles)):
    """ go through the sets as pairs and apply the jaccard function  
    if the similarity is greater than or equal to 0.75 push it to
    the duplicates array"""
    for j_doc in range(i_doc + 1, len(shingles)):
        jaccard_similarity = jaccard(shingles[i_doc], shingles[j_doc])
        is_duplicate = jaccard_similarity >= 0.75
        if is_duplicate:
            
         duplicates.append((i_doc, j_doc, jaccard_similarity))
         
         
jac = pd.DataFrame(duplicates,columns=("Doc 1","Doc2","Simil"))

""" 
create a jac dataFrame ru merge the indexes
"""
dat = dat.drop(dat.index[jac['Doc2']])
"""
Drop the corresponing indexes from the main dataframe dat
"""

""" 
3 Inspect categorical  variables, aburada source amam yrıca diğer categoric variables için de yapılabilir
check what sources we have, burada eğer çok az sayıda olan source varsa çıkarma fonksiyonu eklenebilir"
""
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
    
"""
This is for cleaning the source, add other categories if they are in the metadata, also find something for the date format as well
there are two steps to this: 
"""
def extract_source(source):
    """
    splits the './' etc. from the source column
    alsı the on Sunday and 'The'
    """
    res = re.split('\(|\.|\-',source)[0] 
    res =res.lower()
    res = res.replace('on sunday','') 
    res = res.replace('sunday','') 
    res = re.sub('the\s','',res) 
    res = re.sub('daily\s','',res) 
    out = res.strip() if res else source.strip()
    return(out)

dat['source'] = dat['source'].apply(lambda x: extract_source(x)) 


"""4 Inspect text variables (i.e title, content), min max number"""

"""4.1  özet"""
document_lengths = np.array(list(map(len, df.content.str.split(' '))))
print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))    

""" 4.2  word distribution by doc"""
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax);
"""detecting very short docs"""
print("There are {} documents with over 100 words.".format(sum(document_lengths > 100)))

"""4..3 distribution in shorter docs, this is to decide if any docs should be excluded"""
shorter_documents = document_lengths[document_lengths <= 100]
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(shorter_documents, bins=50, ax=ax);

"""4.4 list of shorter docs, 50 in this case, but change according to the analyses above"""
df[document_lengths <= 50]
""" filter out  shorter docs)"""
df = df[document_lengths > 50]
    


"""5) Feature creation, prporcess the text

The usual approach in natural language processing, is to first kind of cleanse the text. We have to make sure our model can understand similarities and understand when two different words mean similar things. You can't just input the raw text into a model and expect it to understand everything. Therefor, I summarized the most common steps to take in order to achieve this cleansing.
    Remove words that mean little; these are usually words that occur very frequently or words that occur very infrequently. 
    Also, punctuation can be removed, but perhaps you'd like to keep some punctuation in, 
    like exclamation marks or question marks since maybe one writer uses them more than others.
    Stemming; this basically means joining words that mean the same. Take for example the words running and runs, 
    the stem of both words is run. Thus with stemming you'd group these words together and give them the same meaning for the model.
    Vectorize words; since we can't input plain words into a model and expect it to learn from it, we have to vectorize the words. 
    This basically means create unit vectors for all words.
."""

"""5.1 spell corrector", this takes long time to finish, find if there is a faster way"""
from textblob import TextBlob
df['content'][:5].apply(lambda x: str(TextBlob(x).correct()))

""" 5.2 remove non-ascii words"""
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


""" 5.3  text CLEANER, ADD more regexes, more soecific to news"""
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



""" 5.4 Inspect this WC for any extreöme non sense words,  
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


"""5. 5. Prepare Stopwords"""
from spacy.lang.en.stop_words import STOP_WORDS
stop_words = list(STOP_WORDS) # <- set of Spacy's default stop words
stop_words.extend(['imgs', 'syndigate', 'info', 'jpg', 'http', 'photo','eca', 'nd', 'th', 'st', 
                   'system', 'time', 'year', 'people', 'world', 'technology', "telegraph", "quote_component", "emded_component", "float_left", "guardian_media",
                   "daily_mail", "datum", "min_width", 'post_publisher', 'nonascii', "margin_left", "margin_right", "html_embed",
                   "embed_component","network_content", "china_daily", "china_morniing", "copyright_south"])
    
    
    
    
"""MORE TO ADD, lexical diversity, readability, lexical richness etc..."""
"""



"""
Apply the extract_source function to dataframes source column and overwrite it
And finally write it as json
"""
import os
os.chdir ('/home/asuerdem/Documents/ai_culture')

dat.to_json("UK_afterJaccard.json")








