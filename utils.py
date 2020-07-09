import nltk
import pickle
import re
import numpy as np
import csv

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):

    embeddings={}
    with open(embeddings_path,newline='') as embedding_obj:
        lines=csv.reader(embedding_obj,delimiter='\t')
        for line in lines:
            word=line[0]
            embedding=np.array(line[1:]).astype(np.float32)
            embeddings[word]=embedding
        dim=len(line)-1
    return embeddings,dim

def question_to_vec(question, embeddings, dim):
    
    word_embedding=[embeddings[word] for word in question.split() if word in embeddings]
    if not word_embedding:
        return np.zeros(dim)
    words_embeddings = np.array(word_embedding)
    return np.mean(words_embeddings,axis=0)
    
def unpickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
