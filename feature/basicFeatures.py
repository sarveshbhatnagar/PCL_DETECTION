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

    def add_vectors_multiple(self, list_of_sentences, wv):
        """
        Returns sum of word vectors, use on text feature (ref preprocessing)
        params:
        list_of_sentences: list of sentences [["word1", "word2", ...], ["word1", "word2", ...]]
        wv: word2vec model
        """
        ls = []
        if(not list_of_sentences):
            # TODO better handling.
            return self.add_vectors(["the", "is", "a"], wv)
        for i in list_of_sentences:
            ls.append(self.add_vectors(i, wv))
        return sum(ls)

    def get_text_feature(self, text_tokens, n=[3, 5, 7]):
        """
        NOTE: NEW
        """
        nt = []
        for i in range(len(text_tokens)-n[0]):
            for j in n:
                nt.append(text_tokens[i:i+j])

        return nt
