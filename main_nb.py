# Made by Sarvesh Bhatnagar
from dont_patronize_me import DontPatronizeMe


# Misc
import numpy as np

# Preprocess library
import contractions


# Sampling
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# Sklearn Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


# Accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def contract_words(text):
    """
    Removes Contractations from text. i.e. are'nt -> are not
    """
    return contractions.fix(text)


def preprocess_text(text):
    """
    Should return a list of words
    """
    text = str(text)
    text = contract_words(text)
    text = text.lower()
    # text = text.replace(".", " ").replace(",", " ").replace("!", " ").replace(";", " ").replace('"', "").replace("'", "")
    text = text.replace('"', "").replace(
        ",", "").replace("'", "").replace("-", "").replace(";", "")
    return text


if __name__ == '__main__':
    dpm = DontPatronizeMe('dataset', 'dontpatronizeme_pcl.tsv')
    dpm.load_task1()
    data = dpm.train_task1_df
    # data['text'] = data['text'].apply(contract_words)
    data['text'] = data['text'].apply(preprocess_text)

    # Initialize the RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    # Initialize the RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], stratify=data['label'], test_size=0.2, random_state=1)

    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 1)
    x_ros, y_ros = ros.fit_resample(X_train, y_train)
    x_rus, y_rus = rus.fit_resample(X_train, y_train)

    clf_one = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    clf_two = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    x_ros = x_ros.reshape((len(x_ros),))
    x_rus = x_rus.reshape((len(x_rus),))

    clf_one.fit(x_rus, y_rus)
    clf_two.fit(x_ros, y_ros)
    predictions_one = clf_one.predict(X_test)
    predictions_two = clf_two.predict(X_test)

    print("")
    print("Predictions with classifier 1: (RandomUnderSampling)")
    print("Accuracy:", accuracy_score(y_test, predictions_one))
    print("Precision:", precision_score(y_test, predictions_one, average=None))
    print("Recall:", recall_score(y_test, predictions_one, average=None))

    print("")
    print("Predictions with classifier 2: (RandomOverSampling)")
    print("Accuracy:", accuracy_score(y_test, predictions_two))
    print("Precision:", precision_score(y_test, predictions_two, average=None))
    print("Recall:", recall_score(y_test, predictions_two, average=None))
    print("")
