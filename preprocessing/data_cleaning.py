# Author Dhia Rzig
import  pandas as pd
import os
import string
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
dataset_path='..\dataset'
def convert_data_to_csv():
    rows=[]
    rows_no_punct=[]
    rows_no_redundant=[]
    with open(os.path.join(dataset_path, 'dontpatronizeme_pcl.tsv')) as f:
        for line in f.readlines()[4:]:
            par_id=line.strip().split('\t')[0]
            art_id = line.strip().split('\t')[1]
            keyword=line.strip().split('\t')[2]
            country=line.strip().split('\t')[3]
            t=line.strip().split('\t')[4].lower()
            t1=t.translate(t.maketrans('', '', string.punctuation))
            l=line.strip().split('\t')[-1]
            if l=='0' or l=='1':
                lbin=0
            else:
                lbin=1
            # produce csv with all the entries
            rows.append(
                {'par_id':par_id,
                 'art_id':art_id,
                 'keyword':keyword,
                 'country':country,
                 'text':t,
                 'label':lbin,
                 'orig_label':l
                 }
            )
            # produce csv with all the entries but no punctuation
            rows_no_punct.append(
                {'par_id': par_id,
                 'art_id': art_id,
                 'keyword': keyword,
                 'country': country,
                 'text': t1,
                 'label': lbin,
                 'orig_label': l
                 }
            )
            similar_row_exists=False
            # produce csv with only the shorter entries
            t_index=-1
            for row in rows_no_redundant:
                t_index +=1
                if (row['text'] in t) and ( len(t) > len(row['text']) ) and (row['label'] == lbin):
                    similar_row_exists=True
                    break
                elif( t in row['text'] )   and ( len(t) < len(row['text']) ) and (row['label'] == lbin):
                    rows_no_redundant.remove(row)
                    break
            if not similar_row_exists:
                rows_no_redundant.insert(t_index,{
                    'par_id': par_id,
                    'art_id': art_id,
                    'keyword': keyword,
                    'country': country,
                    'text': t,
                    'label': lbin,
                    'orig_label': l
                })
    df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
    df.to_csv('..\dataset\dontpatronizeme_pcl.csv',index = False)
    df_no_punct= pd.DataFrame(rows_no_punct, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
    df_no_punct.to_csv('..\dataset\dontpatronizeme_pcl_no_punct.csv',index = False)
    df_no_redundant = pd.DataFrame(rows_no_redundant,
                               columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label'])
    df_no_redundant.to_csv('..\dataset\dontpatronizeme_pcl_no_redundant.csv',index = False)



def convert_categories_to_csvs( return_one_hot=True):
    tag2id = {
        'Unbalanced_power_relations': 0,
        'Shallow_solution': 1,
        'Presupposition': 2,
        'Authority_voice': 3,
        'Metaphors': 4,
        'Compassion': 5,
        'The_poorer_the_merrier': 6
    }
    data = defaultdict(list)
    with open (os.path.join(dataset_path, 'dontpatronizeme_categories.tsv')) as f:
        for line in f.readlines()[4:]:
            par_id=line.strip().split('\t')[0]
            art_id = line.strip().split('\t')[1]
            text=line.split('\t')[2].lower()
            keyword=line.split('\t')[3]
            country=line.split('\t')[4]
            start=line.split('\t')[5]
            finish=line.split('\t')[6]
            text_span=line.split('\t')[7]
            label=line.strip().split('\t')[-2]
            num_annotators=line.strip().split('\t')[-1]
            labelid = tag2id[label]
            if not labelid in data[(par_id, art_id, text, keyword, country)]:
                data[(par_id,art_id, text, keyword, country)].append(labelid)

    par_ids=[]
    art_ids=[]
    pars=[]
    pars_no_punct=[]
    pars_no_redundant=[]
    keywords=[]
    countries=[]
    labels=[]

    for label in data.values():
        labels.append(label)

    all_index=-1
    for par_id, art_id, par, kw, co in data.keys():
        all_index+=1
        par_ids.append(par_id)
        art_ids.append(art_id)
        pars.append(par)
        pars_no_punct.append(par.translate(par.maketrans('', '', string.punctuation)))
        keywords.append(kw)
        countries.append(co)
        t_index=-1
        similar_row_exists = False
        for par_t in pars_no_redundant:
            t_index+=1
            if (par in par_t) and (len(par_t) > len(par)) and (labels[all_index] == labels[t_index]):
                similar_row_exists = True
                break
            elif (par in par_t) and (len(par_t)  < len(par)) and (labels[all_index] == labels[t_index]):
                pars_no_redundant.remove(par_t)
                break
        if not similar_row_exists:
            pars_no_redundant.insert(t_index,par)


    if return_one_hot:
        labels = MultiLabelBinarizer().fit_transform(labels)

    list_0=[label[0] for label in labels]
    list_1=[label[1] for label in labels]
    list_2=[label[2] for label in labels]
    list_3=[label[3] for label in labels]
    list_4=[label[4] for label in labels]
    list_5=[label[5] for label in labels]
    list_6=[label[6] for label in labels]

    df_sep_labels = pd.DataFrame(list(zip(par_ids,
                                art_ids,
                                pars,
                                keywords,
                                countries,
                                list_0,list_1,list_2,list_3,list_4,list_5,list_6
                               )), columns=['par_id',
                                                    'art_id',
                                                    'text',
                                                    'keyword',
                                                    'country',
                                            'Unbalanced_power_relations',
                                            'Shallow_solution',
                                                'Presupposition',
                                                'Authority_voice',
                                                'Metaphors',
                                                'Compassion',
                                                'The_poorer_the_merrier'
                                                    ])
    df_combined = pd.DataFrame(list(zip(par_ids,
                               art_ids,
                               pars,
                               keywords,
                               countries,
                               labels)), columns=['par_id',
                                                  'art_id',
                                                  'text',
                                                  'keyword',
                                                  'country',
                                                  'label',
                                                  ])
    df_combined_no_punct = pd.DataFrame(list(zip(par_ids,
                                        art_ids,
                                        pars_no_punct,
                                        keywords,
                                        countries,
                                        labels)), columns=['par_id',
                                                           'art_id',
                                                           'text',
                                                           'keyword',
                                                           'country',
                                                           'label',
                                                           ])
    df_combined_no_redundant = pd.DataFrame(list(zip(par_ids,
                                                 art_ids,
                                                 pars_no_redundant,
                                                 keywords,
                                                 countries,
                                                 labels)), columns=['par_id',
                                                                    'art_id',
                                                                    'text',
                                                                    'keyword',
                                                                    'country',
                                                                    'label',
                                                                    ])
    df_sep_labels.to_csv('..\dataset\dontpatronizeme_categories_seperate_labels.csv', index=False)
    df_combined.to_csv('..\dataset\dontpatronizeme_categories_combined_labels.csv',index = False)
    df_combined_no_punct.to_csv('..\dataset\dontpatronizeme_categories_combined_labels_no_punct.csv', index = False)
    df_combined_no_redundant.to_csv('..\dataset\dontpatronizeme_categories_combined_labels_no_redundant.csv',index = False)

# def load_test(self):
#     #self.test_df = [line.strip() for line in open(self.test_path)]
#     rows=[]
#     with open(self.test_path) as f:
#         for line in f.readlines()[4:]:
#             t=line.strip().split('\t')[3].lower()
#             rows.append(t)
#     self.test_set = rows

convert_data_to_csv()
convert_categories_to_csvs()
