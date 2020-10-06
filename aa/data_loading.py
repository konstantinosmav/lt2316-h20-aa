
#basics
import random
import pandas as pd
from numpy.random import choice
import random
from pathlib import Path
import pandas as pd
import torch
from glob import glob
import xml.etree.ElementTree as ET
import re
import os
import string 
import re
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from venn import venn

from collections import Counter
from sklearn.model_selection import train_test_split

#device = torch.device('cuda:1')

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!
    
    
    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.i2w[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.i2n[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        
        super().__init__(data_dir=data_dir, device=device)


    def instantiate_df(self,dir_list): 
        print("reading")
        
        
        self.id2word = defaultdict(int) # map unique token to unique integer
        self.id2ner = defaultdict(int) #map unique label to unique integer
        co_token = 1 # initiate counter
        co_ner = 1
        
        
        self.list_for_token_df = []
        self.list_for_ner_df = []
        
        # go through all subdirectories
        #for subdir, dirs, files in os.walk(dir_list):
        #    #print(files)
        #    for file in files:
        #        if file.endswith(".xml"):
        #            f = os.path.join(subdir, file)
        #            
        #            # create split
        for f in dir_list:
            if "Train" in str(f):
                split = choice(["train","development"],p=[0.8,0.2])
                #print(split)
            else: 
                split = "test"
                
        

            tree = ET.parse(f)
            root = tree.getroot()
            
            #self.max_l = []
            for sentence in root:
                sent_id = sentence.attrib["id"]
                # remove all punctuation to not face problems when tokenizing (i dont want e.g. "paracetamol,")
                sent_text = sentence.attrib["text"]
                
                tokenized_text = sent_text.split(' ')
                #self.max_l.append(len(tokenized_text))
              
                k = 0
                for word in tokenized_text:
                    
                    word=word.replace(',','_') 
                    word=word.replace('.','_')
                    word=word.replace("'","_")
                    #print(word)
                    char_start = k
                    
                    char_end = k + len(word)-1
                   
                    if word.count('_') >= 1:
                        
                        nr_punc = word.count('_')
                        char_end -= nr_punc
                       
                    k += len(word) +1                          
                                         
                    if not self.id2word[word]:
                        self.id2word[word] = co_token
                        co_token +=1
                    
                    
                    token_id = self.get_id_only(word, self.id2word)
                    # append a list with sent_id,token_id,ch_s,ch_e,split on the ult list, every small list will be a row in the token_data_frame 
                    self.list_for_token_df.append([sent_id,token_id,char_start,char_end,split])
        
                for child in sentence:
                    if child.tag == "entity":
                        d_name = child.attrib["text"]
                        ner_id = child.attrib["type"]
                        if not self.id2ner[ner_id]:
                            self.id2ner[ner_id] = co_ner
                            co_ner +=1
                            #print(ner_id)
                        ner_id = self.get_id_only(ner_id,self.id2ner)
                        offset = child.attrib["charOffset"]
                        d_ch = child.attrib["charOffset"].split("-")
                        # for entities that include just one "-"
                        if offset.count('-') == 1: 
                            ch_start,ch_end = d_ch[0],d_ch[1]
                            ch_start,ch_end = int(ch_start), int(ch_end)
                            self.list_for_ner_df.append([sent_id,ner_id,ch_start,ch_end])
                            
                        # for entities that have two "-" in the ch_offset and are separated by ";"    
                        else:
                            for ch_offset in offset.split(";"):
                                ch_start,ch_end = ch_offset.split("-")[0],ch_offset.split("-")[1]
                                ch_start,ch_end = int(ch_start),int(ch_end)
                                self.list_for_ner_df.append([sent_id,ner_id,ch_start,ch_end])
                            
        self.data_df = pd.DataFrame(self.list_for_token_df, columns=['sentence_id','token_id','char_start_id','char_end_id','split'])
        self.ner_df = pd.DataFrame(self.list_for_ner_df, columns=['sentence_id','ner_id','char_start_id','char_end_id'])
        pass
    
                            
        

        
    def get_id_only(self,token,dic):
        #self.token = token
        #self.dic = dic
        for words, ids in dic.items():
            if words == token:
                return ids  
            
            
    def padding(self, sequence):
    
        padded = pad_sequences(sequence,padding='post',value=0)
        return padded
    


    def _parse_data(self,data_dir):
        
        
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        
        
        
        dir_list = []
        for subdir, dirs, files in os.walk(data_dir):
        ##print(files)
            for file in files:
                if file.endswith(".xml"):
                    f = os.path.join(subdir, file)
                    dir_list.append(f)
 
        self.instantiate_df(dir_list)
        self.vocab = list(self.id2word.keys())
        self.max_sample_length = self.get_max_len()
        self.i2w = {v:k for k,v in self.id2word.items()}
        self.i2n = {v:k for k,v in self.id2ner.items()}
      
       
        
        pass
    
    

    def get_labels(self,df):
        # returns the sentence and the labels given to every token of the sentence
        
        sentences = []
        labels = []
        for sent_id in list(set(df["sentence_id"])):
            token_sent = df[df['sentence_id']==sent_id]
            #print(token_sent)
            ner_sent = self.ner_df[self.ner_df['sentence_id']==sent_id]
            #sentence = []
            # maybe i dont need sentence
            label = []
            for i,t_row in token_sent.iterrows():
                #sentence.append(t_row["token_id"])
                is_ner = False
                #print(sentence)
                for i, l_row in ner_sent.iterrows():
                    if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                        label.append(l_row["ner_id"])
                        #print(label)
                        is_ner = True
                if not is_ner:
                    label.append(0)
                    
            #print('hi',sentence,label,len(sentence))        
            #sentences.append(sentence)
            labels.append(label)       
        return labels#sentences,labels
    
    
    def get_max_len(self):        
        return Counter(self.data_df[['sentence_id']]['sentence_id']).most_common(1)[0][1]
    
    
    def get_y(self):
        
        
        #device = torch.device('cuda:0')
        # create the train,test and development dataframes
        df_train = self.data_df.loc[self.data_df['split'] == 'train']
        df_test = self.data_df.loc[self.data_df['split'] == 'test']
        df_dev = self.data_df.loc[self.data_df['split']=='development']
         
        #get labels
        self.train_labels = self.get_labels(df_train)
        self.test_labels = self.get_labels(df_test)
        self.dev_labels =  self.get_labels(df_dev)
        
        #padding stage so that all labels have the same length, 0 is non-ner, -1 is padding
        
        
        self.padded_train = self.padding(self.train_labels)
        self.padded_test = self.padding(self.test_labels)
        self.padded_dev = self.padding(self.dev_labels)
        
        device = torch.device('cuda:0')
        #create tensor
        self.train_tensor = torch.LongTensor(self.padded_train).to(device)
        self.test_tensor = torch.LongTensor(self.padded_test).to(device)
        self.dev_tensor = torch.LongTensor(self.padded_dev).to(device)
        
        return self.train_tensor, self.test_tensor, self.dev_tensor
        



    def plot_split_ner_distribution(self):
        df_train = self.data_df.loc[self.data_df['split'] == 'train']
        df_test = self.data_df.loc[self.data_df['split'] == 'test']
        df_dev = self.data_df.loc[self.data_df['split']=='development']
        
        self.train_labels = self.get_labels(df_train)
        self.test_labels = self.get_labels(df_test)
        self.dev_labels =  self.get_labels(df_dev)
      
        #flatten out lists and don't include 0s i.e. non-ner tokens.
        train_labels_count = Counter([l for sublist in self.train_labels for l in sublist if l != 0])
        test_labels_count = Counter([l for sublist in self.test_labels for l in sublist if l != 0])
        dev_labels_count = Counter([l for sublist in self.dev_labels for l in sublist if l != 0])
        
        data = [train_labels_count,test_labels_count,dev_labels_count]
        
        #plt.hist(train_labels_count, bins = 120, label='train', color = 'b')
        #plt.hist(test_labels_count,bins=120, labek = 'test',color = 'y')
        #plt.hist(dev_labels_count, bins=120, label = 'dev', color = 'g')
                 
        
        df = pd.DataFrame(data, index=['train', 'test', 'dev'])
        #label_t = Counter([l for sublist in labels for l in sublist if l != 0])
        #df =pd.DataFrame([train,test,dev],index=['train',])
        df.plot(kind='bar')
        
                  
        plt.show()
        pass
 
  

    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
      
        df_train = self.data_df[self.data_df['split']=='train']
        df_test = self.data_df[self.data_df['split']=='test']
        df_dev = self.data_df[self.data_df['split']=='development']
    
        sequence_1 = []
        sequence_2 = []
        sequence_3 = []
        
        for sent_id in set(df_train.sentence_id):
            sent = df_train[df_train['sentence_id']==sent_id].values.tolist()
            sequence_1.append(len(sent))
        for sent_id in set(df_test.sentence_id):
            sent = df_test[df_test['sentence_id']==sent_id].values.tolist()
            sequence_2.append(len(sent))
        for sent_id in set(df_dev.sentence_id):
            sent = df_dev[df_dev['sentence_id']==sent_id].values.tolist()
            sequence_3.append(len(sent))
            
        plt.hist(sequence_1, bins=120,label="train", color='b')
        plt.hist(sequence_2, bins = 120, label = "test", color = 'y')
        plt.hist(sequence_3, bins = 120, label = 'dev', color = 'r')
        
        plt.xlabel('Sample length')
        plt.ylabel('Number of sentences')
        plt.show()
        pass
    # FOR BONUS PART!!
 
    def plot_ner_per_sample_distribution(self):        
    # Should plot a histogram displaying the distribution of number of NERs in sentences
    # e.g. how many sentences has 1 ner, 2 ner and so on
        sequence_1 = []    
        
        for sent_id in set(self.ner_df.sentence_id):
            sent = self.ner_df[self.ner_df['sentence_id']==sent_id].values.tolist()
            sequence_1.append(len(sent))
       
        plt.hist(sequence_1, bins=100,label="train", color='y')  
        
        plt.xlabel('Number of ners in sentence')
        plt.ylabel('Number of sentences containing number of ners')
        plt.show()
        pass

        
 
 
 

    def plot_ner_cooccurence_venndiagram(self):
         # FOR BONUS PART!!
         # Should plot a ven-diagram displaying how the ner labels co-occur
        venn_list = []  
        
        for ner_id in self.ner_df['ner_id'].unique():
            ner = self.ner_df[self.ner_df['ner_id']==ner_id]
            sents = ner['sentence_id'].unique().tolist()
            #venn_dic[ner[ner_id]] = set(sents)
            venn_list.append((ner_id,sents))
        #print(venn_list)
        dic= {}
        for v_list in venn_list:
            dic[v_list[0]] = set(v_list[1])
        venn(dic)
        plt.show()
        pass
            
        
 


#basics
#import random
#import pandas as pd
#from numpy.random import choice
#import random
#from pathlib import Path
#import nltk
#import pandas as pd
#import torch
#from glob import glob
#import xml.etree.ElementTree as ET
#import re
#import os
#import string 
#import re
#from collections import defaultdict
#from tqdm import tqdm
#import matplotlib.pyplot as plt
#from keras.preprocessing.sequence import pad_sequences
#import tokenizations
#from nltk import TreebankWordTokenizer
#
#from collections import Counter
#from sklearn.model_selection import train_test_split
#
##device = torch.device('cuda:1')
#
#class DataLoaderBase:
#
#    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!
#    
#    
#    def __init__(self, data_dir:str, device=None):
#        self._parse_data(data_dir)
#        assert list(self.data_df.columns) == [
#                                                "sentence_id",
#                                                "token_id",
#                                                "char_start_id",
#                                                "char_end_id",
#                                                "split"
#                                                ]
#
#        assert list(self.ner_df.columns) == [
#                                                "sentence_id",
#                                                "ner_id",
#                                                "char_start_id",
#                                                "char_end_id",
#                                                ]
#        self.device = device
#        
#
#    def get_random_sample(self):
#        # DO NOT TOUCH THIS
#        # simply picks a random sample from the dataset, labels and formats it.
#        # Meant to be used as a naive check to see if the data looks ok
#        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
#        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
#        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]
#
#        decode_word = lambda x: self.i2w[x]
#        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)
#
#        sample = ""
#        for i,t_row in sample_tokens.iterrows():
#
#            is_ner = False
#            for i, l_row in sample_ners.iterrows():
#                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
#                    sample += f'{self.i2n[l_row["ner_id"]].upper()}:{t_row["token"]} '
#                    is_ner = True
#            
#            if not is_ner:
#                sample += t_row["token"] + " "
#
#        return sample.rstrip()
#
#
#
#class DataLoader(DataLoaderBase):
#
#
#    def __init__(self, data_dir:str, device=None):
#        
#        super().__init__(data_dir=data_dir, device=device)
#
#
#    def instantiate_df(self,dir_list): 
#        print("reading")
#        
#        
#        self.id2word = defaultdict(int) # map unique token to unique integer
#        self.id2ner = defaultdict(int) #map unique label to unique integer
#        co_token = 1 # initiate counter
#        co_ner = 1
#        
#        punctuation = '-,.?!:;.&"'
#        self.list_for_token_df = []
#        self.list_for_ner_df = []
#        
#        # go through all subdirectories
#        #for subdir, dirs, files in os.walk(dir_list):
#        #    #print(files)
#        #    for file in files:
#        #        if file.endswith(".xml"):
#        #            f = os.path.join(subdir, file)
#        #            
#        #            # create split
#        for f in dir_list:
#            if "Train" in str(f):
#                split = choice(["train","development"],p=[0.8,0.2])
#                #print(split)
#            else: 
#                split = "test"
#                
#        
#
#            tree = ET.parse(f)
#            root = tree.getroot()
#            
#            #self.max_l = []
#            for sentence in root:
#                sent_id = sentence.attrib["id"]
#                # remove all punctuation to not face problems when tokenizing (i dont want e.g. "paracetamol,")
#                sent_text = sentence.attrib["text"]
#                if sent_text == "": # to exclude completely empty sentences i e DDI-DrugBank.d228.s4 in Train/DrugBank/Fomepizole_ddi.xml
#                    continue
#                #sent_text = ''.join(i for i in sent_text if not i in punctuation)
#                spans = TreebankWordTokenizer().span_tokenize(sent_text)
#        
#                tokenized_text = TreebankWordTokenizer().tokenize(sent_text)
#                #spans=tokenizations.get_original_spans(tokenized_text, sent_text)
#                for i in range(len(tokenized_text)):
#                    char_start = spans[i][0]
#                    char_end = spans[i][1]-1 #to not get the character end as the space 
#                #self.max_l.append(len(tokenized_text))
#              
#                #k = 0
#                    for word in tokenized_text:
#                        
#                        #word=word.replace(',','_') 
#                        #word=word.replace('.','_')
#                        #word=word.replace("'","_")
#                        ##print(word)
#                        #char_start = k
#                        #
#                        #char_end = k + len(word)-1
#                       #
#                        #if word.count('_') >= 1:
#                        #    
#                        #    nr_punc = word.count('_')
#                        #    char_end -= nr_punc
#                        #   
#                        #k += len(word) +1                          
#                                             
#                        if not self.id2word[word]:
#                            self.id2word[word] = co_token
#                            co_token +=1
#                        
#                        
#                        token_id = self.get_id_only(word, self.id2word)
#                        # append a list with sent_id,token_id,ch_s,ch_e,split on the ult list, every small list will be a row in the token_data_frame 
#                        self.list_for_token_df.append([sent_id,token_id,char_start,char_end,split])
#        
#                for child in sentence:
#                    if child.tag == "entity":
#                        d_name = child.attrib["text"]
#                        ner_id = child.attrib["type"]
#                        if not self.id2ner[ner_id]:
#                            self.id2ner[ner_id] = co_ner
#                            co_ner +=1
#                            #print(ner_id)
#                        ner_id = self.get_id_only(ner_id,self.id2ner)
#                        offset = child.attrib["charOffset"]
#                        d_ch = child.attrib["charOffset"].split("-")
#                        # for entities that include just one "-"
#                        if offset.count('-') == 1: 
#                            ch_start,ch_end = d_ch[0],d_ch[1]
#                            ch_start,ch_end = int(ch_start), int(ch_end)
#                            self.list_for_ner_df.append([sent_id,ner_id,ch_start,ch_end])
#                            
#                        # for entities that have two "-" in the ch_offset and are separated by ";"    
#                        else:
#                            for ch_offset in offset.split(";"):
#                                ch_start,ch_end = ch_offset.split("-")[0],ch_offset.split("-")[1]
#                                ch_start,ch_end = int(ch_start),int(ch_end)
#                                self.list_for_ner_df.append([sent_id,ner_id,ch_start,ch_end])
#                            
#        self.data_df = pd.DataFrame(self.list_for_token_df, columns=['sentence_id','token_id','char_start_id','char_end_id','split'])
#        self.ner_df = pd.DataFrame(self.list_for_ner_df, columns=['sentence_id','ner_id','char_start_id','char_end_id'])
#        pass
#    
#                            
#        
#
#        
#    def get_id_only(self,token,dic):
#        #self.token = token
#        #self.dic = dic
#        for words, ids in dic.items():
#            if words == token:
#                return ids  
#            
#            
#    def padding(self, sequence):
#    
#        padded = pad_sequences(sequence,padding='post',value=0)
#        return padded
#    
#
#
#    def _parse_data(self,data_dir):
#        
#        
#        # Should parse data in the data_dir, create two dataframes with the format specified in
#        # __init__(), and set all the variables so that run.ipynb run as it is.
#        #
#        # NOTE! I strongly suggest that you create multiple functions for taking care
#        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
#        # identify the seperate functions needed.
#        
#        
#        
#        dir_list = []
#        for subdir, dirs, files in os.walk(data_dir):
#        ##print(files)
#            for file in files:
#                if file.endswith(".xml"):
#                    f = os.path.join(subdir, file)
#                    dir_list.append(f)
# 
#        self.instantiate_df(dir_list)
#        self.vocab = list(self.id2word.keys())
#        self.max_sample_length = self.get_max_len()
#        self.i2w = {v:k for k,v in self.id2word.items()}
#        self.i2n = {v:k for k,v in self.id2ner.items()}
#      
#       
#        
#        pass
#    
#    
#
#    def get_labels(self,df):
#        # returns the sentence and the labels given to every token of the sentence
#        
#        sentences = []
#        labels = []
#        for sent_id in list(set(df["sentence_id"])):
#            token_sent = df[df['sentence_id']==sent_id]
#            #print(token_sent)
#            ner_sent = self.ner_df[self.ner_df['sentence_id']==sent_id]
#            #sentence = []
#            # maybe i dont need sentence
#            label = []
#            for i,t_row in token_sent.iterrows():
#                #sentence.append(t_row["token_id"])
#                is_ner = False
#                #print(sentence)
#                for i, l_row in ner_sent.iterrows():
#                    if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
#                        label.append(l_row["ner_id"])
#                        #print(label)
#                        is_ner = True
#                if not is_ner:
#                    label.append(0)
#                    
#            #print('hi',sentence,label,len(sentence))        
#            #sentences.append(sentence)
#            labels.append(label)       
#        return labels#sentences,labels
#    
#    
#    def get_max_len(self):        
#        return Counter(self.data_df[['sentence_id']]['sentence_id']).most_common(1)[0][1]
#    
#    
#    def get_y(self):
#        
#        
#        #device = torch.device('cuda:0')
#        # create the train,test and development dataframes
#        df_train = self.data_df.loc[self.data_df['split'] == 'train']
#        df_test = self.data_df.loc[self.data_df['split'] == 'test']
#        df_dev = self.data_df.loc[self.data_df['split']=='development']
#         
#        #get labels
#        self.train_labels = self.get_labels(df_train)
#        self.test_labels = self.get_labels(df_test)
#        self.dev_labels =  self.get_labels(df_dev)
#        
#        #padding stage so that all labels have the same length, 0 is non-ner, -1 is padding
#        
#        
#        self.padded_train = self.padding(self.train_labels)
#        self.padded_test = self.padding(self.test_labels)
#        self.padded_dev = self.padding(self.dev_labels)
#        
#        device = torch.device('cuda:0')
#        #create tensor
#        self.train_tensor = torch.LongTensor(self.padded_train).to(device)
#        self.test_tensor = torch.LongTensor(self.padded_test).to(device)
#        self.dev_tensor = torch.LongTensor(self.padded_dev).to(device)
#        
#        return self.train_tensor, self.test_tensor, self.dev_tensor
#        
#
#
#
#    def plot_split_ner_distribution(self):
#        df_train = self.data_df.loc[self.data_df['split'] == 'train']
#        df_test = self.data_df.loc[self.data_df['split'] == 'test']
#        df_dev = self.data_df.loc[self.data_df['split']=='development']
#        
#        self.train_labels = self.get_labels(df_train)
#        self.test_labels = self.get_labels(df_test)
#        self.dev_labels =  self.get_labels(df_dev)
#      
#        #flatten out lists and don't include 0s i.e. non-ner tokens.
#        train_labels_count = Counter([l for sublist in self.train_labels for l in sublist if l != 0])
#        test_labels_count = Counter([l for sublist in self.test_labels for l in sublist if l != 0])
#        dev_labels_count = Counter([l for sublist in self.dev_labels for l in sublist if l != 0])
#        
#        data = [train_labels_count,test_labels_count,dev_labels_count]
#        
#        
#        df = pd.DataFrame(data, index=['train', 'val', 'test'])
#        #label_t = Counter([l for sublist in labels for l in sublist if l != 0])
#        #df =pd.DataFrame([train,test,dev],index=['train',])
#        df.plot(kind='bar')
#        
#                  
#        plt.show()
#        pass
# 
#  
#
#    def plot_sample_length_distribution(self):
#         # FOR BONUS PART!!
#         # Should plot a histogram displaying the distribution of sample lengths in number tokens
#         
#        pass
# 
# 
#    def plot_ner_per_sample_distribution(self):        
#         # FOR BONUS PART!!
#         # Should plot a histogram displaying the distribution of number of NERs in sentences
#         # e.g. how many sentences has 1 ner, 2 ner and so on
#        pass
# 
# 
#    def plot_ner_cooccurence_venndiagram(self):
#         # FOR BONUS PART!!
#         # Should plot a ven-diagram displaying how the ner labels co-occur
#         
#        pass
# 
#        
#    
#    
#                                                   
#     