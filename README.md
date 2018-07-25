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
It starts w preprocessing of texts, tokenizing with bigrams and trigrams, cerates a dictionary and corpus w BOW, LDA model, claibrating the performnace of different LDA models,  Visualize the topics-keywords, H optimal number of topics for LDA, Finding the dominant topic in each sentence, find the most representative document for each topic, Topic distribution across documents

original code is here :https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#1introduction



