from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
# from gensim.test.utils import common_texts
from dont_patronize_me import DontPatronizeMe
import contractions
from tensorflow import keras

# Feature
import feature.basicFeatures as bf
import feature.makeWordVector as mwv

# Preprocessing
import preprocessing.basicPreProcessing as bp

# Model
import models.deepModel as dm


def contract_words(text):
    """
    Removes Contractations from text. i.e. are'nt -> are not
    """
    return contractions.fix(text)


def preprocess_text(text):
    """
    Should return a list of words
    """
    text = contract_words(text)
    text = text.lower()
    text = text.replace('"', "").replace(",", "").replace("'", "")
    return text.split()


dpm = DontPatronizeMe('dataset', 'dontpatronizeme_pcl.tsv')



data = dpm.load_task1()
process = bp.BasicPreProcessing()

data['text_split'] = data['text'].apply(preprocess_text)


# model = Word2Vec(sentences=data['text_split'], size=100,
#                  window=5, min_count=1, workers=4)
# model.save('word2vec.model')


# Training the model.
# model = Word2Vec.load("word2vec.model")
# model.train(data['text_split'], total_examples= model.corpus_count, epochs=10)

# Saving KeyedVectors
# word_vectors = model.wv
# word_vectors.save("word2vec.wordvectors")
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')


def get_bigrams(text_split):
    """
    Returns a list of bigrams
    """
    bigrams = [[text_split[i], text_split[i+1]]
               for i in range(len(text_split)-1)]
    return bigrams


def apply_addition(list_of_words):
    """
    Returns the sum of the word vectors
    """
    return sum([wv[word] for word in list_of_words])


data['embeddings'] = data['text_split'].apply(apply_addition)


# rus = RandomUnderSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    data['embeddings'], data['label'], stratify=data['label'], test_size=0.2, random_state=1)
X_train = np.array(X_train)
X_train = X_train.reshape(-1, 1)

x_rus, y_rus = rus.fit_resample(X_train, y_train)

# Below code won't work.
tweet_clf_one = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# x_rus = x_rus.reshape((len(x_rus),))


def create_baseline():
    # create model
    inputs = keras.Input(shape=(100, ), name="digits")
    x = layers.Dense(64, activation=keras.layers.LeakyReLU(
        alpha=0.01), name="dense_1")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.01)(x)
    x = layers.Dense(32, activation=keras.layers.LeakyReLU(
        alpha=0.01), name="dense_3")(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(32, activation=keras.layers.LeakyReLU(
        alpha=0.01), name="dense_4")(x)
    x = layers.Dropout(0.05)(x)
    z = layers.Dense(12, activation="relu", name="dense_5")(x)
    y = layers.Dense(6, activation="relu", name="dense_6")(x)
    x = layers.Concatenate()([z, y])
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_baseline()
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# x_rus = [np.asarray(i).astype(np.float32) for i in x_rus]
x_rus = [item[0] for item in x_rus]
# x_rus[0][0].shape

x_rus = np.array(x_rus).astype(np.float32)

print("Fit model on training data")
history = model.fit(
    x_rus,
    y_rus,
    batch_size=64,
    epochs=150,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    # validation_data=(x_val, y_val),
)


def ready_data(X, y):
    X = np.array(X)
    X = X.reshape(-1, 1)
    x_rus = X
    y_rus = y
    # rus = RandomUnderSampler(random_state=42)
    # x_rus, y_rus = rus.fit_resample(X, y)
    x_rus = [item[0] for item in x_rus]
    x_rus = np.array(x_rus).astype(np.float32)
    return x_rus, y_rus


X_test, y_test = ready_data(X_test, y_test)
predictions = model.predict(X_test)
predictions = [item.argmax() for item in predictions]
y_test = list(y_test)
print(accuracy_score(y_test, predictions))

print(precision_score(y_test, predictions, average=None))

print(recall_score(y_test, predictions, labels=[0, 1], average=None))
