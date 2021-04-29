import cnn_string_preprocess
import pandas as pd

preprocess = cnn_string_preprocess.string_Preprocess()
def custom_string_preprocess(str1):
    return preprocess.remove_special_chars(str1,"twitter")

def yield_data():
    df=pd.read_csv("./davidson_dataset.csv")
    #df=pd.read_csv("founta_dataset_full.csv")
    df = df.drop(['noisy_labels','phrase'],axis=1)
    #df['class'][df['class']=='normal'] = 0
    #df['class'][df['class']=='abusive'] = 1
    #df['class'][df['class']=='hateful'] = 2
    #df = df[df['class']!='spam']


    df["tweet"] = df['tweet'].astype(str).str.lower()
    df["tweet"] = df['tweet'].replace(' +', ' ', regex=True)
    df["tweet"] = df['tweet'].apply(custom_string_preprocess)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df["tweet"],df["class"],test_size=0.2,random_state=1)
    X_train, X_val,  y_train, y_val  = train_test_split(df["tweet"],df["class"],test_size=0.25, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test

#yield_data()
