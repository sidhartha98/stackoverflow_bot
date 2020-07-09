#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cpickle as pickle
# import pickle
import re
import utils



# TF-IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(X_train, X_test, vectorizer_path):

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_test=tfidf_vectorizer.transform(X_test)
    with open(vectorizer_path,'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer,vectorizer_file)
    
    return X_train, X_test


# READING OF DATA
sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)


# PREPROCESSING OF TEXT
from utils import RESOURCE_PATH

dialogue_df['text'] = dialogue_df['text'].apply(text_prepare)
stackoverflow_df['title'] = stackoverflow_df['title'].apply(text_prepare)


# INTENT-RECOGNIZER
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

intent_recognizer=LogisticRegression(solver='newton-cg',C=10, penalty='l2',n_jobs=-1)
intent_recognizer.fit(X_train_tfidf, y_train)
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))
pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb')) #INTENT-RECOGNISER MODEL DUMPED 



# TAG-CLASSIFIER / PROGRAMMING LANGUAGE CLASSIFICATION
X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)

from sklearn.multiclass import OneVsRestClassifier

lr = LogisticRegression(solver='newton-cg',C=5, penalty='l2',n_jobs=-1)
tag_classifier = OneVsRestClassifier(lr)
tag_classifier.fit(X_train_tfidf, y_train)

y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))



# CREATE PICKLE FILE FOR EACH PROGRAMMING LANGUAGE AND STORE IT IN A DIRECTORY TO REDUCE OVERHEAD.
word_embeddings, embeddings_dim = load_embeddings('data/word_embeddings.tsv')
posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
counts_by_tag = posts_df.groupby(posts_df['tag']).count()
counts_by_tag.items()
counts_by_tag = posts_df['tag'].value_counts().to_dict()

import os
os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]
    
    tag_post_ids = tag_posts['post_id'].values
    
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = question_to_vec(title,word_embeddings, embeddings_dim)

    filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))