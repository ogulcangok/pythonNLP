#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:09:25 2018

@author: asuerdem
"""

import pandas as pd
import re

dat = pd.read_json('/home/asuerdem/Documents/ai_culture/UK.json')
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
"""
Apply the extract_source function to dataframes source column and overwrite it
And finally write it as json
"""
import os
os.chdir ('/home/asuerdem/Documents/ai_culture')

dat.to_json("UK_afterJaccard.json")








