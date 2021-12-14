
import numpy as np
import pandas as pd

import contractions
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
import itertools, re


from sklearn.naive_bayes import CategoricalNB, ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report


import warnings

warnings.filterwarnings("ignore")


def load_task2():
    temp = open('dontpatronizeme_categories.tsv')
    check = temp.read()
    tag2id = {'Unbalanced_power_relations': 0, 'Shallow_solution': 1, 'Presupposition': 2, 'Authority_voice': 3,
              'Metaphors': 4,
              'Compassion': 5, 'The_poorer_the_merrier': 6}
    ls = check.split('\n')[4:]
    df = pd.DataFrame([line.split('\t') for line in ls if len(line) > 0])
    df.columns = ['par_id', 'art_id', 'text', 'keyword', 'country', 'start', 'finish', 'text_span', 'label',
                  'num_annotators']
    df['labelid'] = df['label'].map(tag2id)
    return df



pcl_cat_df = load_task2()

pcl_cat_df.labelid.value_counts() / pcl_cat_df.shape[0]


# %% md
### PREPROCESSING
# %%
def contract_words(text):
    """
    Removes contractions from the text.
    """
    return contractions.fix(text)


def preprocess_text(text):
    """
    Should return a list of words
    """
    text = contract_words(text)
    text = text.lower()
    #     text = text.replace('"', "").replace(",", "").replace("'", "")
    text = text.replace('"', "").replace(",", "").replace("'", "").replace(".", " .")  ## added by PAVAN

    ## To capture multiple # feature -- added by PAVAN
    if re.search(r'[a-z]+\#', text):
        tmp_ls = text.split()
        text = ' '.join(
            [re.sub(pattern=r'\#', repl=' #', string=str(i)) if re.search(r'[a-z]+\#', str(i)) else i for i in tmp_ls])

    ## To capture # feature -- added by PAVAN
    if re.search(r'\#[a-z]+', text):
        tmp_ls = text.split()
        text = ' '.join(
            [re.sub(pattern=r'\#', repl='hashtagfea ', string=str(i)) if re.search(r'\#[a-z]+', str(i)) else i for i in
             tmp_ls])
    return text.split()

preprocess_text('#thedignityproject')
preprocess_text('#ichh#thedignityproject')


# %% md
## FEATURE GENERATION - word2vec
# %%

class Word2VecModelTrainer:
    """
    Word2Vec Features. Basic Process: call train, call load.
    """

    def __init__(self, sentences=[], path="word2vec.wordvectors"):
        """
        params:
        sentences: list of sentences (default: [])
        NOTE sentences should be like this
        [["i", "am", "ready"],["some","other","sentence"]]
        """

        self.sentences = sentences
        self.path = path

    ## UPDATED version -- PAVAN
    def train(self, size=100, window=7):
        """
        Trains the model on sentences
        params:
        size: size of the vector
        window: window size
        """

        model = Word2Vec(sentences=self.sentences, vector_size=size,
                         window=window, min_count=1, workers=4)
        model.save("word2vec_categories.model")

        model.build_vocab(corpus_iterable=self.sentences, update=True)

        model.train(corpus_iterable=self.sentences,
                    total_examples=model.corpus_count, epochs=30)

        #         model.train(sentences=self.sentences,
        #                     total_examples=model.corpus_count, epochs=30)
        word_vectors = model.wv
        word_vectors.save(self.path)
        return model, word_vectors

    def load_trained(self, path=""):
        """
        Loads the trained model
        """
        if (path != ""):
            load_path = path
        else:
            load_path = self.path
        wv = KeyedVectors.load(load_path, mmap='r')
        return wv


# %%
## Features
def get_bigrams(text_split):
    """
    :param text_split: split text (ref preprocessing).
    :return: list of bigrams
    """
    bigrams = [[text_split[i], text_split[i + 1]]
               for i in range(len(text_split) - 1)]

    return bigrams


def add_vectors(list_of_words, wv):
    """
    Returns sum of word vectors, use on text split (ref preprocessing)
    params:
    list_of_words: list of words
    wv: word2vec model
    """
    return sum([wv[word] for word in list_of_words])


# %%
## For creating word2vec model & embeddings -> I am using 'text' column. Since it has all the data
pcl_cat_df['text_split'] = pcl_cat_df['text'].apply(preprocess_text)

unique_sentences_ls = pcl_cat_df['text_split'].to_list()
# # Train WordVectors. Only run once.
Word2VecModelTrainer(sentences=unique_sentences_ls, path="word2vec_categories.wordvectors").train()

# %%
# Load the trained word vectors.
wv = Word2VecModelTrainer().load_trained("word2vec_categories.wordvectors")

pcl_cat_df['text_span_split'] = pcl_cat_df['text_span'].apply(preprocess_text)


## Get the treatment dictionary
def get_treatment_map(TEXT_SPAN_SPLIT_SERIES, TEXT_SPLIT_SERIES, EMBED):
    def custom_key(str):
        return -len(str), str.lower()

    text_span_token_ls = list(
        set(list(itertools.chain.from_iterable(TEXT_SPAN_SPLIT_SERIES))))  # pcl_cat_df.text_span_split
    text_token_ls = list(set(list(itertools.chain.from_iterable(TEXT_SPLIT_SERIES))))  # pcl_cat_df.text_split
    TOBE_TREATED = [token for token in text_span_token_ls if token not in text_token_ls]

    sorted_token_ls = sorted(sorted(EMBED.key_to_index.keys()), key=custom_key)
    EMBED_sorted = {i: EMBED[i] for i in sorted_token_ls}
    TREATING_TOKENS_MAP = {i: j for i in TOBE_TREATED for j in EMBED_sorted if re.match(pattern=i, string=j)}
    return TREATING_TOKENS_MAP


TREATING_TOKENS_MAP = get_treatment_map(pcl_cat_df.text_span_split, pcl_cat_df.text_split, wv)

## Applying the Treatment on "text_span_split" column
pcl_cat_df['text_span_split'] = pcl_cat_df['text_span_split'].apply(
    lambda x: [TREATING_TOKENS_MAP[i] if i in TREATING_TOKENS_MAP else i for i in x])

pcl_cat_df['embeddings'] = pcl_cat_df['text_span_split'].apply(add_vectors, wv=wv)  # basic_features.

# %% md
## MODEL VERSION 1 - word2vec- NaiveBayes
# %%
## Use 'labelid' column for using CategoricalNB


X_train, X_test, y_train, y_test = train_test_split(
    pcl_cat_df['embeddings'], pcl_cat_df['labelid'], stratify=pcl_cat_df['labelid'], test_size=0.2, random_state=1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(np.array(list(X_train)))
X_test = scaler.fit_transform(np.array(list(X_test)))

model_NB = ComplementNB()
model_NB.fit(X_train, y_train)
pred_NB = model_NB.predict(X_test)
# %%
print(classification_report(y_true=y_test, y_pred=pred_NB))
# %% md
### Feature Generation: TF-IDF
# %%
corpus = pcl_cat_df.text_split.apply(lambda x: ' '.join(x)).drop_duplicates().to_list()

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
X = vectorizer.transform(pcl_cat_df.text_span)
# %%
pcl_textspan_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
pca = PCA(n_components=100, random_state=123)

pcl_cat_df['tfidf_embeddings'] = list(list(pca.fit_transform(pcl_textspan_tfidf)))

# %% md
## MODEL VERSION 2 - tfidf- NaiveBayes
# %%
X_train, X_test, y_train, y_test = train_test_split(
    pcl_cat_df['tfidf_embeddings'], pcl_cat_df['labelid'], stratify=pcl_cat_df['labelid'], test_size=0.2,
    random_state=1)

# define oversampling strategy
rus = RandomOverSampler(sampling_strategy='minority', random_state=42)  #
# fit and apply the transform
X_over, y_over = rus.fit_resample(np.array(list(X_train)), np.array(list(y_train)))

X_train = X_over.copy()
y_train = y_over.copy()

# y_train = y_train.sort_values()
# SORTED_INDICES= list(y_train.sort_values().index)
# X_train = X_train.loc[SORTED_INDICES]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(np.array(list(X_train)))
X_test = scaler.fit_transform(np.array(list(X_test)))

model_NB = ComplementNB()
model_NB.fit(X_train, y_train)
pred_NB = model_NB.predict(X_test)

print(classification_report(y_true=y_test, y_pred=pred_NB))
# %%

# %%

# %%

# %%

# %%

# %%

# %%
## Preprocessing examples

## Points to be noted:
## 892 index: Text : #thedignityproject text_span = thedignityproject
## -> To capture the essence of hashtag. I will replace #abc -> hashtagfea abc & get the hashtagfea into textspan so that the embedding of hashtag is present

## Text : selected. text_span = selected
## -> To tackle this replaced '.' with ' . '

## hashtag example
pcl_cat_df.loc[
    (pcl_cat_df['text_span_split'].apply(lambda x: ' '.join(x)).str.find('thedignityproject') != -1), 'text'].to_list()
pcl_cat_df.loc[(pcl_cat_df['text_span_split'].apply(lambda x: ' '.join(x)).str.find(
    'thedignityproject') != -1), 'text_split'].to_list()

## merged hashtag example
pcl_cat_df.loc[pcl_cat_df['text_split'].apply(lambda x: 'hashtagfea' in x), 'text'].to_list()

## irregular clip - futur example
pcl_cat_df.loc[(pcl_cat_df['text_span_split'].apply(lambda x: 'futur' in x)), 'text_span']
# %%
