# Made by Sarvesh Bhatnagar


class BasicFeatures:
    def __init__(self):
        pass

    def get_bigrams(self, text_split):
        """
        :param text_split: split text (ref preprocessing).
        :return: list of bigrams
        """
        bigrams = [[text_split[i], text_split[i+1]]
                   for i in range(len(text_split)-1)]

        return bigrams

    def add_vectors(self, list_of_words, wv):
        """
        Returns sum of word vectors, use on text split (ref preprocessing)
        params:
        list_of_words: list of words
        wv: word2vec model
        """
        return sum([wv[word] for word in list_of_words])
