import cnn_string_preprocess
import pandas as pd

preprocess = cnn_string_preprocess.string_Preprocess()
def custom_string_preprocess(str1):
    return preprocess.remove_special_chars(str1,"twitter")

def yield_data():
    df=pd.read_csv("./hateval2019_en_train.csv")
    
    df = df.drop(['TR','AG'],axis=1)

    df["text"] = df['text'].astype(str).str.lower()
    df["text"] = df['text'].replace(' +', ' ', regex=True)
    df["text"] = df['text'].apply(custom_string_preprocess)
    X_train=df["text"]
    y_train=df["HS"]
    
    df=pd.read_csv("./hateval2019_en_dev.csv")
    
    df = df.drop(['TR','AG'],axis=1)

    df["text"] = df['text'].astype(str).str.lower()
    df["text"] = df['text'].replace(' +', ' ', regex=True)
    df["text"] = df['text'].apply(custom_string_preprocess)
    X_val=df["text"]
    y_val=df["HS"]
    
    df=pd.read_csv("./hateval2019_en_test.csv")
    
    df = df.drop(['TR','AG'],axis=1)

    df["text"] = df['text'].astype(str).str.lower()
    df["text"] = df['text'].replace(' +', ' ', regex=True)
    df["text"] = df['text'].apply(custom_string_preprocess)
    X_test=df["text"]
    y_dev=df["HS"]

    
    return X_train, X_val, X_test, y_train, y_val, y_test

#yield_data()
