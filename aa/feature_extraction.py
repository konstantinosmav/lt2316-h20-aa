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

def embed_feat(df,id2word,embedding_dim):
    print("extracting features")    
    sentences = get_padded_sents(df)
    print('done making sentences')
    all_feat = []    
    embeddings = nn.Embedding((len(id2word)+1), embedding_dim)
    for sent in sentences:
        f_sent = []
        for token_id in sent:
            # .squeeze() to remove all 1s
            embed = embeddings(torch.LongTensor([token_id])).squeeze()            
            f_sent.append(embed)
        all_feat.append(torch.stack(f_sent))
    tensor_feat = torch.stack(all_feat)    
    return tensor_feat           

def extract_features(data:pd.DataFrame,id2word,embedding_dim,device):
    df_train = data.loc[data['split'] == 'train']
    df_test = data.loc[data['split'] == 'test']
    df_dev = data.loc[data['split']=='development']         
    
    X_train = embed_feat(df_train,id2word,embedding_dim).to(device)
    X_test = embed_feat(df_test,id2word,embedding_dim).to(device)
    X_dev = embed_feat(df_dev,id2word,embedding_dim).to(device)
    
    return X_train, X_test, X_dev

