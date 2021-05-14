import cnn_string_preprocess
import pandas as pd

preprocess = cnn_string_preprocess.string_Preprocess()
def custom_string_preprocess(str1):
    return preprocess.remove_special_chars(str1,"twitter")

def yield_data(test_program=False):
    df=pd.read_csv("./founta_dataset_full.csv")
    #df=pd.read_csv("founta_dataset_full.csv")
    df = df.drop(['noisy_labels','phrase'],axis=1)
    #print(df.head()) 
    ## drop spam
    df['class'][df['class']=='normal'] = 0
    df['class'][df['class']=='abusive'] = 1
    df['class'][df['class']=='hateful'] = 2
    df = df[df['class']!='spam']

    #df['class'][df['class']==2]=1
    #df['class'][df['class']==3]=2

    df["sentence"] = df['sentence'].astype(str).str.lower()
    df["sentence"] = df['sentence'].replace(' +', ' ', regex=True)
    df["sentence"] = df['sentence'].apply(custom_string_preprocess)
    #print(df.head()) 
    train = df.sample(frac=0.70, random_state=0)
    val = df.drop(train.index)
    test = val.sample(frac=0.66, random_state=0)
    val = val.drop(test.index)

    train.to_csv('splitted_datasets/train.tsv',index=true,header=True,sep='\t')
    val.to_csv('splitted_datasets/val.tsv',index=true,header=True,sep='\t')
    test.to_csv('splitted_datasets/test.tsv',index=true,header=True,sep='\t')
    #return train, val, test
   
#train_data, dev_data,test_data = yield_data(test_program=False)
#print(train_data['sentence'].values)
yield_data()
