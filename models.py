import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
import gensim.downloader
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer

# download a pretrained Word2Vec model
# 'glove-wiki-gigaword-300' (376.1 MB)
# 'word2vec-ruscorpora-300' (198.8 MB)
# 'word2vec-google-news-300' (1.6 GB)

vectorizer = gensim.downloader.load('word2vec-ruscorpora-300')

# create a corpus of tokens
corpus = documents['context'].tolist()    ## to change
corpus = [preprocess_string(t) for t in corpus]

# defining parameters of embedding model
embedding_configs = {
    'sentences' : corpus, 
    'vector_size' : 300, 
    'window' : 5, 
    'min_count' : 1, 
    'workers' : 4
}

# defining the TF-IDF
tfidf_configs = {
    'lowercase': True,
    'analyzer': 'word',
    'stop_words': 'english',
    'binary': True,
    'max_df': 0.9,
    'max_features': 10_000
}

# defining the number of documents to retrieve
retriever_configs = {
    'n_neighbors': 1,
    'metric': 'cosine'
}


class tf_idf_retriever:
    def __init__(self):
    # defining our pipeline
        self.embedding = TfidfVectorizer(**tfidf_configs)
        self.retriever = NearestNeighbors(**retriever_configs)

    def fit(self, contexts):
        ''' Fit context embeddings to context ids
            Input : dict/dataframe in format {'context','c_id'}
        '''
        # let's train the model to retrieve the document id 'c_id'
        X = self.embedding.fit_transform(contexts['context'])
        self.retriever.fit(X, contexts['c_id'])

    def predict(self, question):
        ''' Predict predict k best contexts for each question
            Input : list/Series of str objects (questions)
            Output : array of context ids of shape len(question)*k 
        '''
        X = self.embedding.transform(question)
        y_pred = self.retriever.kneighbors(X, return_distance=False)
        return y_pred
    
    def vectorized(self, text): 
        ''' Return vectorized version of text, using the model vectorizer'''
        vector = self.embedding.transform([text])
        return self.embedding.inverse_transform(vector)



class word2vec_retriever: 
    def __init__(self):
    # defining our pipeline
        self.embedding = Word2Vec(**embedding_configs).wv
        self.retriever = NearestNeighbors(**retriever_configs)

    def vectorized(self, text, verbose=False):
        '''
        Transform the text in a vector[Word2Vec]
        vectorizer: sklearn.vectorizer
        text: str
        '''
        tokens = preprocess_string(text)
        words = [self.embedding[w] for w in tokens if w in self.embedding]
        if verbose:
            print('Text:', text)
            print('Vector:', [w for w in tokens if w in  self.embedding])
        elif len(words):
            return np.mean(words, axis=0)
        else:
            return np.zeros((300), dtype=np.float32)

    def fit(self, contexts):
        X = contexts['context'].apply(self.vectorized).tolist()
        self.retriever.fit(X, contexts['c_id'])

    def predict(self, question):
        # vectorizer the questions
        X = question.apply(self.vectorized).tolist()

        # predict one document for each question
        y_pred = self.retriever.kneighbors(X, return_distance=False)

        return y_pred