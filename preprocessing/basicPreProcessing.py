# Input Text, Output Text.
# Made by Sarvesh Bhatnagar
import contractions


class BasicPreProcessing:
    def __init__(self, text):
        """
        Takes in text and preprocesses it.
        Going with design pattern that no function will modify
        the text. but instead return a new text. self modification 
        is discouraged.
        param text: input text
        """
        self.text = text

    def preprocess(self, isSplit=False):
        """
        Main Function to do preprocessing.
        param isSplit: if true, splits the text into words.
        """
        text = self.text
        text = self.remove_contractions(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        if isSplit:
            return text.split()
        return text

    def remove_punctuation(self, text):
        """
        Removes punctuation from the text.
        Currently removes .,?!;"'
        NOTE high scope for optimization.

        param text: input text
        """
        return text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").replace(";", " ").replace('"', "").replace("'", "")

    def remove_stopwords(self, text):
        """
        Removes stopwords from the text.
        """

        return text

    def remove_contractions(self, text):
        """
        Removes contractions from the text.
        """
        return contractions.fix(text)