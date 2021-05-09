# Copyright (c) Microsoft. All rights reserved.
from random import shuffle
from data_utils.metrics import calc_metrics

import mt_dnn_string_preprocess
import pandas as pd

preprocess = mt_dnn_string_preprocess.string_Preprocess()
def custom_string_preprocess(str1):
    return preprocess.remove_special_chars(str1,"twitter")

def load_hateval(file, header=True, is_train=True):
    rows = []
    cnt = 0
    print("preprocessing hateval dataset.............")
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split(',')
            if is_train and len(blocks) < 2: continue
            lab = 0
            if is_train:
                lab = int(blocks[1])
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
                print("Current hateval sample: ",sample)
            else:
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows
    
    
    
def load_davidson(dataset):
    
        df=pd.read_csv(dataset)  

        df["tweet"] = df['tweet'].astype(str).str.lower() 
        df["tweet"] = df['tweet'].replace(' +', ' ', regex=True) 
        df["tweet"] = df['tweet'].apply(custom_string_preprocess)
        print("Preprocessing davidson dataset.......") 

        processed_dataset = [] 
        for index, row in df.iterrows():
            sample = {'uid': index, 'premise': row['tweet'], 'label': row['class']}
            print("Current davidson sample: ",sample) 
            processed_dataset.append(sample) 
        print("data preprocessed successfully!") 
        return processed_dataset
    
def load_waseem(dataset):
    
        df=pd.read_csv(dataset)  
        df = df.drop(['ids'],axis=1)
        df['class'][df['class']=='none'] = 0
        df['class'][df['class']=='racism'] = 1  
        df['class'][df['class']=='sexism'] = 2 

        df["sentence"] = df['sentence'].astype(str).str.lower() 
        df["sentence"] = df['sentence'].replace(' +', ' ', regex=True) 
        df["sentence"] = df['sentence'].apply(custom_string_preprocess)
        print("Preprocessing waseem dataset.......") 

        processed_dataset = [] 
        for index, row in df.iterrows():
            sample = {'uid': index, 'premise': row['sentence'], 'label': row['class']}
            print("Current waseem sample: ",sample) 
            processed_dataset.append(sample) 
        print("data preprocessed successfully!") 
        return processed_dataset

def load_founta(dataset):

        df=pd.read_csv(dataset)
        #df = df.drop(['noisy_labels','phrase'],axis=1)
        df = df.drop(['ids'],axis=1)
        df['class'][df['class']=='normal'] = 0
        df['class'][df['class']=='abusive'] = 1
        df['class'][df['class']=='hateful'] = 2
        df = df[df['class']!='spam']

        df["sentence"] = df['sentence'].astype(str).str.lower()
        df["sentence"] = df['sentence'].replace(' +', ' ', regex=True)
        df["sentence"] = df['sentence'].apply(custom_string_preprocess)
    
        print("Preprocessing founta dataset.......")
        
        processed_dataset = []
        for index, row in df.iterrows():
            sample = {'uid': index, 'premise': row['sentence'], 'label': row['class']}
            print("Current founta sample: ",sample) 
            processed_dataset.append(sample)
        print("data preprocessed successfully!")
        return processed_dataset

def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))

