#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:39:12 2018

@author: asuerdem
"""


# more common imports
import pandas as pd
import numpy as np
from collections import Counter
import re

# languange processing imports
import nltk
from gensim.corpora import Dictionary
# preprocessing imports
from sklearn.preprocessing import LabelEncoder

# model imports
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV

# visualization imports
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


"""1) FEATURE INSPECTION 
This exploratory phase is always a good thing to start with to get an idea of the data and 
it will probably help you out making decisions later on in the process of 
any data science problem you have and in particular now for text classification.
"""
"""check if there's missing data"""
df.isnull().sum()

"""Inspect source variable"""
"""check what sources we have"""
df.source.value_counts().index
"""barchart of sources"""
fig, ax = plt.subplots(1,1,figsize=(8,6))
source_vc = df.source.value_counts()
ax.bar(range(7), source_vc)
ax.set_xticks(range(7))
ax.set_xticklabels(source_vc.index, fontsize=16)
for rect, c, value in zip(ax.patches, ['b', 'r', 'g', 'y', 'c', 'm'], source_vc.values):
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.5*height, value, ha='center', va='center', fontsize=10, color='white')

"""Inspect text variables, min max number"""

document_lengths = np.array(list(map(len, df.content.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))    
""" word distribution by doc"""
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(document_lengths, bins=50, ax=ax);

"""detecting very short docs"""
print("There are {} documents with over 100 words.".format(sum(document_lengths > 100)))

"""distribution in shorter docs"""
shorter_documents = document_lengths[document_lengths <= 100]
fig, ax = plt.subplots(figsize=(15,6))
ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
sns.distplot(shorter_documents, bins=50, ax=ax);
"""list of shorter docs, 50 in this case, but change according to the analyses above"""
sd = df[document_lengths <= 100]
""" DO the same for lon ger docs and ADD somth to reomve the shorter and longer docs"""


cleansed_words_df = pd.DataFrame.from_dict(id2word.token2id, orient='index')
cleansed_words_df.rename(columns={0: 'id'}, inplace=True)

cleansed_words_df['freq'] = list(map(lambda id_: id2word.dfs.get(id_), cleansed_words_df.id))
del cleansed_words_df['id']

cleansed_words_df.sort_values('freq', ascending=False, inplace=True)
ax = word_frequency_barplot(cleansed_words_df)
ax.set_title("Document Frequencies (Number of documents a word appears in)", fontsize=16);



"""add lexical diversity, richness"""

"""
2) Feature creation

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
"""remove non-ascii words"""
df = df.reset_index(drop=True)
our_special_word = 'qwerty'
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



def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

"""Here we get transform the documents into sentences for the word2vecmodel
# we made a function such that later on when we make the submission, we don't need to write duplicate code"""
def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['content'] = df.content.str.lower()
    df['document_sentences'] = df.content.str.split('.')  # split contents into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists

w2v_preprocessing(df)




def lda_get_good_tokens(df):
    df['content'] = df.content.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.content))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

lda_get_good_tokens(df)


from collections import Counter
tokenized_only_dict = Counter(np.concatenate(df.tokenized_text.values))
tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
tokenized_only_df.rename(columns={0: 'count'}, inplace=True)


""" Barplot of most frequent words"""
# I made a function out of this since I will use it again later on 
def word_frequency_barplot(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))
    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    return ax  
ax = word_frequency_barplot(tokenized_only_df)
ax.set_title("Word Frequencies", fontsize=16);


"""2.2) Remove stopwords that mean little"""

def remove_stopwords(df):
    """ Removes stopwords based on a known set of stopwords
    available in the nltk package. In addition, we include our
    made up word in here.
    """
    # Luckily nltk already has a set of stopwords that we can remove from the texts.
    stopwords = nltk.corpus.stopwords.words('english')
    # we'll add our own special word in here 'qwerty'
    stopwords.append(our_special_word)

    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_text']))

remove_stopwords(df)
"""2.3 stem"""
def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))

stem_words(df)


"""2.4) Vectorize words"""
dictionary = Dictionary(documents=df.stemmed_text.values)

print("Found {} words.".format(len(dictionary.values()))) 



#Make a BOW for every document
def document_to_bow(df):
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
    
document_to_bow(df)


def lda_preprocessing(df):
    """ All the preprocessing steps for LDA are combined in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    lda_get_good_tokens(df)
    remove_stopwords(df)
    stem_words(df)
    document_to_bow(df)
     
"""Visualize the cleansed words frequencies"""


cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
cleansed_words_df.rename(columns={0: 'id'}, inplace=True)

cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
del cleansed_words_df['id']
cleansed_words_df.sort_values('count', ascending=False, inplace=True)        

ax = word_frequency_barplot(cleansed_words_df)
ax.set_title("Document Frequencies (Number of documents a word appears in)", fontsize=16);

""" by source"""
df.source.value_counts().index

tele = list(np.concatenate(df.loc[df.source == 'telegraph', 'stemmed_text'].values))
guard = list(np.concatenate(df.loc[df.source == 'guardian', 'stemmed_text'].values))
ind = list(np.concatenate(df.loc[df.source == 'independent', 'stemmed_text'].values))
times = list(np.concatenate(df.loc[df.source == 'times', 'stemmed_text'].values))
mir = list(np.concatenate(df.loc[df.source == 'mirror', 'stemmed_text'].values))
mail = list(np.concatenate(df.loc[df.source == 'mail', 'stemmed_text'].values))

"""count 50 top words"""
tele_frequencies = {word: tele.count(word) for word in cleansed_words_df.index[:50]}
guard_frequencies = {word: guard.count(word) for word in cleansed_words_df.index[:50]}
ind_frequencies = {word: ind.count(word) for word in cleansed_words_df.index[:50]}
times_frequencies = {word: times.count(word) for word in cleansed_words_df.index[:50]}
mir_frequencies = {word: mir.count(word) for word in cleansed_words_df.index[:50]}
mail_frequencies = {word: mail.count(word) for word in cleansed_words_df.index[:50]}
"""cumulate, why not group_by?"""
frequencies_df = pd.DataFrame(index=cleansed_words_df.index[:50])
frequencies_df['tele_freq'] = list(map(lambda word:
                                      tele_frequencies[word],
                                      frequencies_df.index))
frequencies_df['tel_gua_freq'] = list(map(lambda word:
                                          tele_frequencies[word] + guard_frequencies[word],
                                          frequencies_df.index))
frequencies_df['tel_gua_ind_freq'] = list(map(lambda word:
                                          tele_frequencies[word] + guard_frequencies[word] + ind_frequencies[word],
                                          frequencies_df.index))
frequencies_df['tel_gua_ind_tim_freq'] = list(map(lambda word:
                                          tele_frequencies[word] + guard_frequencies[word] + ind_frequencies[word] + times_frequencies[word],
                                          frequencies_df.index))

    """plot"""
fig, ax = plt.subplots(1,1,figsize=(20,5))
nr_top_words = len(frequencies_df)
nrs = list(range(nr_top_words))
sns.barplot(nrs, frequencies_df['tel_gua_ind_tim_freq'].values, color='c', ax=ax, label="times")
sns.barplot(nrs, frequencies_df['tel_gua_ind_freq'].values, color='b', ax=ax, label="independent")
sns.barplot(nrs, frequencies_df['tel_gua_freq'].values, color='g', ax=ax, label="guardian")
sns.barplot(nrs, frequencies_df['tele_freq'].values, color='r', ax=ax, label="telegraph")

ax.set_title("Word frequencies per source", fontsize=16)
ax.legend(prop={'size': 16})
ax.set_xticks(nrs)
ax.set_xticklabels(frequencies_df.index, fontsize=14, rotation=90);

"""3. LDA"""
corpus = df.bow

num_topics = 150
#A multicore approach to decrease training time
LDAmodel = LdaMulticore(corpus=corpus,
                        id2word=dictionary,
                        num_topics=num_topics,
                        workers=4,
                        chunksize=4000,
                        passes=7,
                        alpha='asymmetric')

def document_to_lda_features(lda_model, document):
    """ Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    """
    topic_importances = LDAmodel.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:,1]

df['lda_features'] = list(map(lambda doc:
                                      document_to_lda_features(LDAmodel, doc),
                                      df.bow))


    
tele_topic_distribution = df.loc[df.source == 'telegraph', 'lda_features'].mean()
guard_topic_distribution = df.loc[df.source == 'guardian', 'lda_features'].mean()
ind_topic_distribution = df.loc[df.source == 'independent', 'lda_features'].mean()
time_topic_distribution = df.loc[df.source == 'times', 'lda_features'].mean()
mir_topic_distribution = df.loc[df.source == 'mirror', 'lda_features'].mean()
mail_topic_distribution = df.loc[df.source == 'mail', 'lda_features'].mean()


fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(30,15))
nr_top_bars = 5
ax1.set_title("Telegraph", fontsize=16)
ax2.set_title("Guardian", fontsize=16)
ax3.set_title("Independent", fontsize=16)
ax3.text(-10, 0.04, "Average Probability of Topic", fontsize=15, ha="center", va="center",
         rotation="vertical")

for ax, distribution, color in zip([ax1,ax2,ax3],
                                   [tele_topic_distribution,guard_topic_distribution,ind_topic_distribution],
                                   ['b','g','r']):
    # Individual distribution barplots
    ax.bar(range(len(distribution)), distribution, alpha=0.7)
    rects = ax.patches
    for i in np.argsort(distribution)[-nr_top_bars:]:
        rects[i].set_color(color)
        rects[i].set_alpha(1)
    # General plotting adjustments
    ax.set_xlim(-1, 150)
    ax.set_xticks(range(20,149,20))
    ax.set_xticklabels(range(20,149,20), fontsize=16)
    ax.set_ylim(0,0.02)
    ax.set_yticks([0,0.01,0.02])
    ax.set_yticklabels([0,0.01,0.02], fontsize=16)

fig.tight_layout(h_pad=3.)

""" see topic words"""
def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
    """ Returns the top words for topic_id from lda_model.
    """
    id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
    word_ids = np.array(id_tuples)[:,0]
    words = map(lambda id_: lda_model.id2word[id_], word_ids)
    return words
for author, distribution in zip(['Tele', 'guard', 'indpnt'], [tele_topic_distribution, guard_topic_distribution, ind_topic_distribution]):
    print("Looking up top words from top topics from {}.".format(author))
    for x in sorted(np.argsort(distribution)[-5:]):
        top_words = get_topic_top_words(LDAmodel, x)
        print("For topic {}, the top words are: {}.".format(x, ", ".join(top_words)))
    print("")

"""4. word2vec"""

sentences = []
for sentence_group in df.tokenized_sentences:
    sentences.extend(sentence_group)

print("Number of sentences: {}.".format(len(sentences)))
print("Number of texts: {}.".format(len(df)))

%%time
# Set values for various parameters
num_features = 200    # Word vector dimensionality
min_word_count = 3    # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
W2Vmodel = Word2Vec(sentences=sentences,
                    sg=1,
                    hs=0,
                    workers=num_workers,
                    size=num_features,
                    min_count=min_word_count,
                    window=context,
                    sample=downsampling,
                    negative=5,
                    iter=6)

"""Word2Vec feature inspection"""

def get_w2v_features(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

df['w2v_features'] = list(map(lambda sen_group:
                                      get_w2v_features(W2Vmodel, sen_group),
                                      df.tokenized_sentences))
    
tele_w2v_distribution = df.loc[df.source == 'telegraph', 'w2v_features'].mean()
guard_w2v_distribution = df.loc[df.source == 'guardian', 'w2v_features'].mean()
ind_w2v_distribution = df.loc[df.source == 'independent', 'w2v_features'].mean()

fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(20,10))

nr_top_bars = 5

ax1.set_title("Telegraph w2v feature distributions", fontsize=16)
ax2.set_title("Guardian w2v feature distributions", fontsize=16)
ax3.set_title("Independent w2v feature distributions", fontsize=16)
ax3.text(-10, 2.3, "Average feature vectors", fontsize=30, ha="center", va="center", rotation="vertical")

for ax, distribution, color in zip([ax1,ax2,ax3], [tele_w2v_distribution,guard_w2v_distribution,ind_w2v_distribution], ['b','g','r']):
    # Individual distribution barplots
    ax.bar(range(len(distribution)), distribution, alpha=0.7)
    rects = ax.patches
    for i in np.argsort(distribution)[-nr_top_bars:]:
        rects[i].set_color(color)
        rects[i].set_alpha(1)
    # General plotting adjustments
    ax.set_xlim(-1, 200)
    ax.set_xticks(range(20,199,20))
    ax.set_xticklabels(range(20,199,20), fontsize=16)
    ax.set_ylim(-0.8,0.8)

fig.tight_layout(h_pad=3.)




"""Classification and hyperparameter tuning"""
label_encoder = LabelEncoder()

label_encoder.fit(df.source)
df['source_id'] = label_encoder.transform(df.source)

df.source_id.value_counts().index




def get_cross_validated_model(model, param_grid, X, y, nr_folds=5):
    """ Trains a model by doing a grid search combined with cross validation.
    args:
        model: your model
        param_grid: dict of parameter values for the grid search
    returns:
        Model trained on entire dataset with hyperparameters chosen from best results in the grid search.
    """
    # train the model (since the evaluation is based on the logloss, we'll use neg_log_loss here)
    grid_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_log_loss', cv=nr_folds, n_jobs=-1, verbose=True)
    best_model = grid_cv.fit(X, y)
    # show top models with parameter values
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    display(result_df[show_columns].sort_values(by='rank_test_score').head())
    return best_model



"""since train_data['lda_features'] and train_data['w2v_features'] don't have the needed shape and type yet,
# we first have to transform every entry"""
X_train_lda = np.array(list(map(np.array, df.lda_features)))
X_train_w2v = np.array(list(map(np.array, df.w2v_features)))
X_train_combined = np.append(X_train_lda, X_train_w2v, axis=1)



from sklearn.utils.testing import all_estimators
estimators = all_estimators()
for name, class in all_estimators: if hasattr(class, 'predict_proba'): print(name)


# store all models in a dictionary
models = dict()
# LDA features only
lr = LogisticRegression()

param_grid = {'penalty': ['l1', 'l2']}
best_lr_lda = get_cross_validated_model(lr, param_grid, X_train_lda, df.source_id)

models['best_lr_lda'] = best_lr_lda


# Word2Vec features only
lr = LogisticRegression()

param_grid = {'penalty': ['l1', 'l2']}

best_lr_w2v = get_cross_validated_model(lr, param_grid, X_train_w2v, df.source_id)

models['best_lr_w2v'] = best_lr_w2v



topla=[]
for e in q:
 idx=e[0]
 liste=e[1]
 for e2 in liste:
  print([idx,e2[0], e2[1]])
  topla.append([idx,e2[0], e2[1]])

tb=pd.DataFrame(topla)

for i, word in enumerate(words):  
  pprint (word + ":" + str(result[i,2]))




    
    
