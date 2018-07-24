#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:57:09 2018

@author: asuerdem
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:57:09 2018

@author: asuerdem
"""

#####2. 3. Import Packages
# Run in python console
import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""5. Prepare Stopwords
NLTK Stop words"""
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['imgs', 'syndigate', 'info', 'jpg', 'http', 'photo','eca', 'nd', 'th', 'st', 'system', 'time', 'year',
                   'people', 'world', 'technology', 'post_publisher'])


"""6. Import Newsgroups Data"""
# Import Dataset
df = pd.read_json('/home/asuerdem/Documents/ai_culture/UK_afterJaccard.json')
#print(df.source.unique()) #prints news sources

"""CLEANER, ADD more regexes, more soecific to news
Convert content to list, and clean the text with regex"""
data = df.content.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', content) for content in data]
# Remove new line characters
data = [re.sub('\s+', ' ', content) for content in data]
# Remove distracting single quotes
data = [re.sub("\'", "", content) for content in data]

#pprint(data[:1])


"""MAYBE A SENTENCE TOKENIZER"""

"""8. Tokenize words and preprocess the text"""

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))


#print(data_words[:2])


"""9. Creating Bigram and Trigram Models
Build the bigram and trigram models"""
bigram = gensim.models.Phrases(data_words, min_count=2, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
"""See trigram example"""
#print(trigram_mod[bigram_mod[data_words[9]]])

""" b,grams only
d =bigrams.vocab
d.items()
topla=[]
for e in d.items():
 idx=e[0]
 liste=e[1]
 print([idx,e[1]])
 topla.append([idx,e[1]])
tb=pd.DataFrame(topla)

str_df = tb.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
for col in str_df:
    tb[col] = str_df[col]
tb.sort_values(by=[1], ascending=True)
""" 

"""10. Remove Stopwords, Make Bigrams and Lemmatize
Define functions for stopwords, bigrams, trigrams and lemmatization"""
nlp = spacy.load("en")
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

"""Remove Stop Words"""
data_words_nostops = remove_stopwords(data_words)# Remove Stop Words
""" Form Bigrams"""
data_words_bigrams = make_bigrams(data_words_nostops)
#data_words_bi_nostops = remove_stopwords(data_words_bigrams)#this is added for removing bigram SWS

""" Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en"""
nlp = spacy.load('en', disable=['parser', 'ner'])
"""Do lemmatization keeping only noun, adj, vb, adv"""
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN' ]) #, 'VERB', 'ADV', 'ADJ',


""" THIS IS FOR BIGRAM ONLY CORPUS
topla=[]
for idx in range(len(data_lemmatized)):
    for token in bigram[data_lemmatized[idx]]:
        if '_' in token:
          topla.append(idx)
          topla.append(token)
"""      
         

"""11. Create the Dictionary and Corpus needed for Topic Modeling
# Create Dictionary"""
id2word = corpora.Dictionary(data_lemmatized) 

"""size of dictionary"""
print("Found {} words.".format(len(id2word.values())))
"""filter extremes"""
id2word.filter_extremes(no_above=0.90, no_below=2)
id2word.compactify()  # Reindexes the remaining words after filtering
print("Left with {} words.".format(len(id2word.values())))

""" corpus"""  
corpus = [id2word.doc2bow(text) for text in data_lemmatized]
        
         
#print(data_lemmatized[:8])
#print(corpus[:1])
#id2word[5]

# Human readable format of corpus (term-frequency)
x = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:]]

#####Feature extraction: TFIDF

from gensim.models import TfidfModel

tfidf_model = gensim.models.TfidfModel(corpus=corpus,
                                         id2word=id2word)
z = [[(id2word[id], freq) for id, freq in cp] for cp in  tfidf_model[corpus]]



"""####################################################################################################
UPTO NOW CORPUS BUILDING, THIS CAN BE USED FOR EVERYTHING. MUCH BETTER THAN QUANTEDA,
######################################################################################################
#to write the output into csv, ask how to turn this into csv
dat = [pd.DataFrame(el) for el in x]
dat = [pd.DataFrame(fd) for fd in dat]
counter = 0
for fd in dat:
   fd.columns = ['word','freq']
   fd['docid'] = [counter] * fd.shape[0]
   counter += 1
output = pd.concat(dat)
#merge the columns in dataframe
df['docid'] = df.index
output =pd.merge(df, output, on = "docid")
output = output.drop(columns ='content')

import os
os.chdir ('/home/asuerdem/Documents/ai_culture')
output.to_csv('UK_gensim.csv')

 
#####################################################################################################
#FROM NOW ON, LDA, HOWEVER, OTHERS CAN BE INSRTED
######################################################################################################"""

lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=id2word, num_topics=10)
lsi.print_topics(10)


# Build LDA model


LDAmodel = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=100, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# Print the Keyword in the 10 topics
pprint(LDAmodel.print_topics())
doc_lda = lda_model[corpus]

#################################################
#LDA topic inspection
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
                                      corpus))

ind_topic_distribution = df.loc[df.source == 'independent', 'lda_features'].mean()
guard_topic_distribution = df.loc[df.source == 'guardian', 'lda_features'].mean()
mir_topic_distribution = df.loc[df.source == 'mirror', 'lda_features'].mean()  

# visualization imports
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
%matplotlib inline
sns.set()  # defines the style of the plots to be seaborn style

fig, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(10,10))

nr_top_bars = 5

ax1.set_title("IND topic distributions", fontsize=10)
ax2.set_title("GUARD topic distributions", fontsize=10)
ax3.set_title("MIRROR topic distributions", fontsize=10)
ax3.text(-10, 0.04, "Average Probability of Topic", fontsize=30, ha="center", va="center",
         rotation="vertical")

for ax, distribution, color in zip([ax1,ax2,ax3],
                                   [ind_topic_distribution,guard_topic_distribution,mir_topic_distribution],
                                   ['b','g','r']):
    # Individual distribution barplots
    ax.bar(range(len(distribution)), distribution, alpha=0.7)
    rects = ax.patches
    for i in np.argsort(distribution)[-nr_top_bars:]:
        rects[i].set_color(color)
        rects[i].set_alpha(1)
    # General plotting adjustments
    ax.set_xlim(-1, 99)
    ax.set_xticks(range(5,20,5))
    ax.set_xticklabels(range(5,20,5), fontsize=16)
    ax.set_ylim(0,0.02)
    ax.set_yticks([0,0.01,0.02])
    ax.set_yticklabels([0,0.01,0.02], fontsize=16)

fig.tight_layout(h_pad=3.)

def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
    """ Returns the top words for topic_id from lda_model.
    """
    id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
    word_ids = np.array(id_tuples)[:,0]
    words = map(lambda id_: lda_model.id2word[id_], word_ids)
    return words

for source, distribution in zip(['independent', 'guardian', 'mirror'], [ind_topic_distribution,guard_topic_distribution,mir_topic_distribution]):
    print("Looking up top words from top topics from {}.".format(source))
    for x in sorted(np.argsort(distribution)[-5:]):
        top_words = get_topic_top_words(LDAmodel, x)
        print("For topic {}, the top words are: {}.".format(x, ", ".join(top_words)))
    print("")
    

####TF*IDF########################
lda_model = gensim.models.ldamodel.LdaModel(tfidf_model[corpus], id2word=id2word, num_topics=14)
pprint(lda_model.print_topics())
###############################################################

#########14. Compute Model Perplexity and Coherence Score#####

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics NEEDS TO PLOT CANNOT PLOT

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)


#######MALLET#####
#16. Building LDA Mallet Model
mallet_path = '/home/asuerdem/Documents/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=15, id2word=id2word)
# Show Topics
pprint(ldamallet.show_topics(formatted=False))

y =ldamallet.show_topics(formatted=False)


# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


####17. How to find the optimal number of topics for LDA?

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# Select the model and print the topics
optimal_model = model_list[2]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))



####18. Finding the dominant topic in each sentence

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


###19. Find the most representative document for each topic

# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()


#20. Topic distribution across documents
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
freqs
# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics


##############WORD2VEC

from gensim.models import Word2Vec

# Initialize and train the model



# Initialize and train the model
W2Vmodel = Word2Vec(texts,
                    sg=1,
                    hs=0,
                    workers=20,
                    size=200,
                    min_count=3,
                    window=5,
                    sample=1e-3 ,
                    negative=5,
                    iter=50)

W2Vmodel.train(texts, total_examples=len(texts), epochs=10)
W2Vmodel.wv.most_similar("artificial_intelligence")
similar_words = {search_term: [item[0] for item in W2Vmodel.wv.most_similar([search_term], topn=5)]
                  for search_term in ['job', 'government', 'industry', 'automation', 'worker', 'robot', 'work', 'economy','technology']}
similar_words


from sklearn.manifold import TSNE

words = sum([[k] + v for k, v in similar_words.items()], [])
wvs = W2Vmodel.wv[words]

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(14, 8))
plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
    
    





words = list(W2Vmodel.wv.vocab)
W2Vmodel.vector_size

#to save the model
import os
os.chdir ('/home/asuerdem/Documents/ai_culture')
W2Vmodel.save('model.bin')

# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

#PCA
from sklearn.decomposition import PCA
from matplotlib import pyplot
####1st way
word_vectors = np.vstack([W2Vmodel[w] for w in words])
word_vectors.shape
X =np.transpose(word_vectors) #if you do that for the words, dont transpose
X.shape

twodim = PCA().fit_transform(X)[:,:2]
twodim.shape
plt.figure(figsize=(5,5))
plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
for word, (x,y) in zip(words, twodim):
    plt.text(x, y, word)
plt.axis('off');

for word, (x,y) in zip(words, twodim):
    print(x, y, word)


from nltk.cluster import KMeansClusterer
import nltk
NUM_CLUSTERS=10
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)


words = list(W2Vmodel.wv.vocab)
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))
    


#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize




data = ["service customer business bank, consumer, company, product, price, industry",
        "system, computer researcher research ai process knowledge",
        "robot human machine intelligence humanity life idea science_fiction",
        "university science research professor study engineer student school",
        "image	user	video	tool	art	picture	project	team	", 
        "game	computer program	player	deepmind	move	chess	human	alphago",
        "car control research development weapon expert ai threat",
        "story film emotion character man movie day series version",
        "machine	program brain	intelligence	question	mind	thing	person computer",
        "facebook apple	thing	user	internet	phone	question	home	assistant"]




tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]


max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

model= Doc2Vec.load("d2v.model")


similar_doc = model.docvecs.most_similar('2')
print(similar_doc)



from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)  
X = kmeans_model.fit(model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()


l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
datapoint = pca.transform(model.docvecs.doctag_syn0)

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()


from nltk.cluster import KMeansClusterer
import nltk
NUM_CLUSTERS=2
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(model.docvecs.doctag_syn0, assign_clusters=True)
print (assigned_clusters)

