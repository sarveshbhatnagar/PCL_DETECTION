# File will include the working.
# Made by Sarvesh Bhatnagar
# dontpatronizeme
from dont_patronize_me import DontPatronizeMe

# Feature
import feature.basicFeatures as bf
import feature.makeWordVector as mwv

# Preprocessing
import preprocessing.basicPreProcessing as bp

# Model
import models.deepModel as dm

# Misc for model training.
from tensorflow import keras
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import contractions


# Scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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


# Deep Learning Pipeline.
if __name__ == '__main__':
    # Load the data.
    dpm = DontPatronizeMe('dataset', 'dontpatronizeme_pcl.tsv')
    dpm.load_task1()
    data = dpm.train_task1_df
    process = bp.BasicPreProcessing()
    data['text_split'] = data['text'].apply(preprocess_text)

    # Train WordVectors. Only run once.
    # mwv.Word2VecModelTrainer(
    #     sentences=data['text_split'], path="dataword.wordvectors").train()

    # Load the trained word vectors.
    wv = mwv.Word2VecModelTrainer().load_trained("word2vec.wordvectors")

    # Make Embedding Columns for each text split.
    basic_features = bf.BasicFeatures()
    data['embeddings'] = data['text_split'].apply(
        basic_features.add_vectors, wv=wv)

    # Random Under Sampler.
    rus = RandomUnderSampler(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        data['embeddings'], data['label'], stratify=data['label'], test_size=0.2, random_state=1)

    # Initializing NNModel/Deep Learning Model.
    # by default ip 100,0 and op 2 i.e. 2 classes classification.
    nn_model = dm.NNModels()

    # Prepare the data for training and testing.
    rus = RandomUnderSampler(random_state=42)
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 1)
    x_rus, y_rus = rus.fit_resample(X_train, y_train)
    x_rus = [item[0] for item in x_rus]
    x_rus = np.array(x_rus).astype(np.float32)
    # x_train, y_train = nn_model.dl_0_process(
    #     X=X_train, y=y_train, isTrain=True)
    # x_test, y_test = nn_model.dl_0_process(X=X_test, y=y_test, isTrain=False)

    model = nn_model.dl_0()
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    # model = nn_model.dl_0_compile(model)

    # Train the model.
    print("Training the model...")
    history = model.fit(x_rus, y_rus, batch_size=64, epochs=150)

    # Prepare testing data.
    X_test, y_test = ready_data(X_test, y_test)

    predictions = model.predict(X_test)
    predictions = [item.argmax() for item in predictions]
    y_test = list(y_test)
    print("Accuracy", accuracy_score(y_test, predictions))
    print("Precision", precision_score(y_test, predictions, average=None))
    print("Recall", recall_score(
        y_test, predictions, labels=[0, 1], average=None))
