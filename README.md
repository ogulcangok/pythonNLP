# exercises
This repository is for my exercises about text mining
1. dedupANDcleaner.py contains codes for deduplicating and cleaning the metadata before corpus construction
this is where the original code is: https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html

2.  LDA_W2V file contains codes for LDA & Word2Vec with hyperparameter training

    Feature inspection: inspect some basic statistics of the dataset.
    Feature creation:  the preprocessing of the data and the LDA and Word2Vec model will be explained and applied.
    Model training: a grid search can be applied for hyperparameter training.
    Model selection: elect the model and make a submission.

original code is here : https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec
ML part does not work
3. gensim_LDA.py
includes code for preprocessing text, and LDA mallet and gensim
Topic Modeling is a technique to extract the hidden topics from large volumes of text. Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling with excellent implementations in the Python’s Gensim package. The challenge, however, is how to extract good quality of topics that are clear, segregated and meaningful. This depends heavily on the quality of text preprocessing and the strategy of finding the optimal number of topics. This tutorial attempts to tackle both of these problems.

original code is here :https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction

Contents
1. Introduction
2. Prerequisites – Download nltk stopwords and spacy model
3. Import Packages
4. What does LDA do?
5. Prepare Stopwords
6. Import Newsgroups Data
7. Remove emails and newline characters
8. Tokenize words and Clean-up text
9. Creating Bigram and Trigram Models
10. Remove Stopwords, Make Bigrams and Lemmatize
11. Create the Dictionary and Corpus needed for Topic Modeling
12. Building the Topic Model
13. View the topics in LDA model
14. Compute Model Perplexity and Coherence Score
15. Visualize the topics-keywords
16. Building LDA Mallet Model
17. How to find the optimal number of topics for LDA?
18. Finding the dominant topic in each sentence
19. Find the most representative document for each topic
20. Topic distribution across documents

