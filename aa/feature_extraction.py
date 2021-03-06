import pandas as pd
import torch
import torch.nn as nn
from keras.preprocessing.sequence import pad_sequences



def padding(sequence):

    padded = pad_sequences(sequence,padding='post',value=0)
    return padded

def get_padded_sents(df):            
    sentences = []
    for sent_id in set(df["sentence_id"]):
        sentence = []
        token_sent = df[df['sentence_id']==sent_id]        
        for i,t_row in token_sent.iterrows():
            sentence.append(t_row["token_id"])  
        sentences.append(sentence)
    return padding(sentences)

def embed_feat(df):
    print("extracting features")    
    sentences = get_padded_sents(df)
    print('done making sentences')
    all_feat = []    
    
    for sent in sentences:
        f_sent = []
        for token_id in sent:
            
            tok_tens = torch.LongTensor([token_id])           
            f_sent.append(tok_tens)
        all_feat.append(torch.stack(f_sent))
    tensor_feat = torch.stack(all_feat)    
    return tensor_feat               

def extract_features(data:pd.DataFrame,device):
    df_train = data.loc[data['split'] == 'train']
    df_test = data.loc[data['split'] == 'test']
    df_dev = data.loc[data['split']=='development']    
    
    train_tokens = df_train.token_id.unique().tolist()
    test_tokens = df_test.token_id.unique().tolist()    
    dev_tokens = df_dev.token_id.unique().tolist() 
    
    td_tokens = test_tokens + dev_tokens
    unk_tokens = [token for token in td_tokens if token not in train_tokens] 
    
    
    df_test.loc[df_test["token_id"].isin(unk_tokens), "token_id"] = -1
    df_dev.loc[df_dev["token_id"].isin(unk_tokens), "token_id"] = -1
        
    X_train = embed_feat(df_train).to(device)
    X_test = embed_feat(df_test).to(device)
    X_dev = embed_feat(df_dev).to(device)
    
    return X_train, X_test, X_dev


