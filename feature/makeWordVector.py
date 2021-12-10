# Made by Sarvesh Bhatnagar
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


class Word2VecModelTrainer:
    """
    Word2Vec Features. Basic Process: call train, call load.
    """

    def __init__(self, sentences=[]):
        """
        params: 
        sentences: list of sentences (default: [])


        NOTE sentences should be like this 
        [["i", "am", "sarvesh"],["some","other","sentence"]]
        """

        self.sentences = sentences

    def train(self, size=100, window=7):
        """
        Trains the model on sentences
        params: 
        size: size of the vector
        window: window size
        """

        model = Word2Vec(sentences=self.sentences, size=size,
                         window=window, min_count=1, workers=4)
        model.save("word2vec.model")

        model.train(sentences=self.sentences,
                    total_examples=model.corpus_count, epochs=30)
        word_vectors = model.wv
        KeyedVectors.save("word2vec.wordvectors")
        return model, word_vectors

    def load_trained(self):
        """
        Loads the trained model
        """
        wv = KeyedVectors.load("word2vec.wordvectors")
        return wv
