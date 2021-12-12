# Input Text, Output Text.
# Made by Sarvesh Bhatnagar , Dhia Rzig
import contractions
import string

class BasicPreProcessing:
    def __init__(self, text="some default text"):
        """
        Takes in text and preprocesses it.
        Going with design pattern that no function will modify
        the text. but instead return a new text. self modification
        is discouraged.
        param text: input text
        """
        self.text = text

    def preprocess(self, text, isSplit=True):
        """
        Main Function to do preprocessing.
        param isSplit: if true, splits the text into words.
        """
        if text == "":
            text = self.text
        else:
            text = self.text
        text = self.remove_contractions(text)
        text = text.lower()
        text = self.remove_punctuation(text)
        # text = self.remove_stopwords(text)
        if isSplit:
            return text.split()
        return text

    def remove_punctuation(self, t):
        """
        Removes punctuation from the text.
        Currently removes .,?!;"'
        NOTE high scope for optimization.

        param text: input text
        """
        return t.translate(t.maketrans('', '', string.punctuation))
        # return text.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").replace(";", " ").replace('"', "").replace("'", "")

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
