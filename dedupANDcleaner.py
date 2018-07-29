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
df.isnull().sum() #bu na remove gibi bir komutla replace edielecek

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
3. This is for cleaning the source, add other categories if they are in the metadata, also find something for the date format as well
there are two steps to this: 
"""
"""3.1. clean the metadata, categorical variables such as sıource, section etc. 
burada, bir önceki explore your data da bulduklarını girecek, yani çıkarılmasını istediklerini yazacak arayüzde"""

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

""" 3.2 convert date , and extract year and month as separate colmuns, buraya gerkli kod gir"""

"""3.3  clean shorter - longer docs,"""
document_lengths = np.array(list(map(len, df.content.str.split(' '))))
df[document_lengths <= 50]
""" filter out  shorter docs)"""
df = df[document_lengths > 50] # 50 will change according to introduced vaue
""" filter out  longr docs)"""
df = df[document_lengths > 2000] # 2000 will change according to introduced vaue


""""following steps needs to be done
        also, sentence tokenizer, and sentence counter neeeded
    removes empty lines
    removes redundant space characters
    drops lines (and their corresponding lines), that are empty, too short, too long
    """
    

"""
    4. preporcess the text
""

"""4.1 spell corrector", this takes long time to finish, find if there is a faster way, or maybe ommited, or a warning message"""

from textblob import TextBlob
df['content'][:5].apply(lambda x: str(TextBlob(x).correct()))

""" 4. 2 remove non-ascii words"""
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


""" 4.3 text CLEANER, ADD more regexes, more soecific to news"""
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

"""
Apply the extract_source function to dataframes source column and overwrite it
And finally write it as json
"""
import os
os.chdir ('/home/asuerdem/Documents/ai_culture')

dat.to_json("UK_afterJaccard.json")








