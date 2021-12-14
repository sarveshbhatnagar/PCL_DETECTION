import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer


class DontPatronizeMe:

    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path
        self.train_task1_df = None
        self.train_task2_df = None
        self.test_set = None

    def load_task1(self, type=''):
        # type :by default, it loads the reuglar dataset. no_punct will remove all punctuation, no_redudant will remove all redundancies
        if type == '':
            self.train_task1_df = pd.read_csv(os.path.join(
                self.train_path, 'dontpatronizeme_pcl.csv'))
        elif type == 'no_punct':
            self.train_task1_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_pcl_no_punct.csv'))
        elif type == 'no_redundant':
            self.train_task1_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_pcl_no_punct_no_redundant.csv'))
        else:
            raise('Unsupported type parameter')
        return self.train_task1_df

    def load_task2(self, return_one_hot=True, type=''):
        # type :by default, it loads the reuglar dataset. no_punct will remove all punctuation, no_redudant will remove all redundancies,seperate will seperate all the labels into different columns

        if type == '':
            self.train_task2_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_categories_combined_labels.csv'))
        elif type == 'no_punct':
            self.train_task2_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_categories_combined_labels_no_punct.csv'))
        elif type == 'no_redundant':
            self.train_task2_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_categories_combined_labels_no_redundant.csv'))
        elif type == 'seperate':
            self.train_task2_df = pd.read_csv(
                os.path.join(self.train_path, 'dontpatronizeme_categories_seperate_labels.csv'))
        else:
            raise ('Unsupported type parameter')
        return self.train_task2_df

    def load_test(self):
        rows = []
        with open(self.test_path) as f:
            for line in f.readlines()[4:]:
                t = line.strip().split('\t')[3].lower()
                rows.append(t)
        self.test_set = rows
        return self.test_set
