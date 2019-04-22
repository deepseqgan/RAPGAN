
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:04:31 2017

@author: gama
"""
from  keras.utils import multi_gpu_model
import tensorflow as tf

import jieba
import jieba.posseg as pseg
import jieba.analyse
import re
import numpy as np
from keras.models import Model
from keras.layers import Embedding,Masking
from keras.layers import Input, Dense,Reshape,concatenate,Flatten,Activation,Permute,multiply
from keras.layers import GRU,Conv1D,LSTM,MaxPooling1D,GlobalMaxPooling1D,TimeDistributed,RepeatVector
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Lambda,Dropout
from keras.utils import to_categorical,multi_gpu_model
import gc
import random
import nltk

#from pypinyin import pinyin, Style
#import pypinyin
#pinyin('天黑', style=Style.FINALS)
from tqdm import tqdm
import pickle
#from keras import backend as K
import csv
import keras.backend as K
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
K.set_session(sess)
jieba.load_userdict('dict.txt.big.txt')
jieba.load_userdict('NameDict_Ch_v2.txt')

from keras.models import load_model

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
import time
start_time = time.time()
token_stream = []

#pair=10000
que_pad=40
ans_pad=20
stop_count=0
pair_train1=[]
pair_train2=[]
check_stop=[]
mod_check_stop=[]
count=0
def jieba_keywords(news):
    keywords = jieba.analyse.extract_tags(news, topK=6)
    #print(keywords)
    return keywords

with open('data/preprocess_rap.csv', 'r',encoding='utf-8',errors='ignore') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        seg_content2=jieba.cut(row[1].replace('\n',''),cut_all=False)
        s_c2="@".join(seg_content2)
        ans=s_c2.split("@")
        if len(ans)>ans_pad:
            continue
        if len(ans)<7:
            continue
        seg_content=jieba.cut(row[0].replace('\n',''),cut_all=False)
        s_c="@".join(seg_content)
        question=s_c.split("@")
        if len(question)>ans_pad:
            continue
        if len(question)<7:
            continue
        while len(question)<que_pad:
               question.append('PAD')
        token_stream.extend(question)
        pair_train1.append(question)
        seg_content2=jieba.cut(row[1].replace('\n',''),cut_all=False)
        s_c2="@".join(seg_content2)
        ans=s_c2.split("@")
        ans.append('EOS')
        ans_len=len(ans)
        while len(ans)<ans_pad: 
            ans.append('PAD')
        if len(ans)>ans_pad:
            ans_len=19
            ans=ans[:ans_pad]
        k=jieba_keywords(row[1])
        temp=[]
        for key,i in enumerate(ans):
            if i in k:
                check_stop.append(stop_count)
                temp.append(key)
            stop_count+=1
        if ans_len not in temp:
            temp.append(ans_len)
        if 19 not in temp:    
            temp.append(19)    
        check_stop.append(stop_count-1)    
        mod_check_stop.append(temp)              
        token_stream.extend(ans)
        pair_train2.append(ans)
#        if len(pair_train1)>pair:
#            break
        
print(len(pair_train1))
pair=len(pair_train1)
#TOP=['PAD','EOS']             
#TOP.extend(token_stream)         
words =list(set(token_stream))
del token_stream

word2idx = dict((word, i) for i, word in enumerate(words))
num_words = len(words)
print("num_words:")
print(num_words)
          
#with open('wordidx.pickle', 'wb') as handle:
#    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('wordidx.pickle', 'rb') as handle:
#     word2idx = pickle.load(handle)            
print('process_data')

predict_pair=ans_pad



for i in range(len(pair_train1)):
    for j in range(que_pad):
        pair_train1[i][j]=word2idx[pair_train1[i][j]]

for i in range(len(pair_train2)):
    for j in range(ans_pad):
        pair_train2[i][j]=word2idx[pair_train2[i][j]]

train_x=[]
train_y=[]
for i in range(len(pair_train1)):
    train_x.append(pair_train1[i][::-1][:que_pad]) 
    train_y.append([pair_train2[i][0]])
    for j in range(1,ans_pad):
#        if str(pair_train2[i][j]) == str(word2idx['PAD']):
#            break
        train_x.append(pair_train1[i][::-1][j:que_pad]+pair_train2[i][:j]) 
        train_y.append([pair_train2[i][j]])


 
train_x=np.array(train_x)
train_y=np.array(train_y)


# In[2]:


mod_check_stop[2]


# In[3]:


print(train_x.shape)
print(train_y.shape)
def get_model():
    dim=128
    inputs = Input(shape=(que_pad,))
    g_emb=Embedding(num_words+1,dim, input_length=(que_pad))(inputs)
    decoder = GRU(dim)(g_emb)
    decoder = Dense(num_words,activation="softmax")(decoder)
    model = Model(inputs=inputs , outputs=decoder)
    return model




# In[4]:


dim=128
d_input1=Input(shape=(que_pad,))
d_input2=Input(shape=(1,))
#,mask_zero=True
con=concatenate([d_input1,d_input2],axis=1)
d_emb=Embedding(num_words+1,dim, input_length=(que_pad+1))(con)
activations = GRU(dim, return_sequences=True)(d_emb)
attention_weight = TimeDistributed(Dense(1, activation='tanh'))(activations) 
attention_weight = Flatten()(attention_weight)
attention_weight = Activation('softmax')(attention_weight)
attention = RepeatVector(dim)(attention_weight)
attention = Permute([2, 1])(attention)
sent_representation = multiply([activations, attention])
sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(dim,))(sent_representation)
probabilities = Dense(2, activation='softmax')(sent_representation)
discriminator = Model([d_input1,d_input2] , probabilities)
discriminator.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
attention_model=Model([d_input1,d_input2] , attention_weight)


# In[5]:


import math

def ppx(y_true, y_pred):
     loss = K.sparse_categorical_crossentropy(y_true, y_pred)
     perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
     return perplexity
   

g_model=get_model()
#sampling_model= multi_gpu_model(get_model(), gpus=2)
g_model.compile(loss=ppx, optimizer='adam',metrics=['accuracy'])
earlyStopping=EarlyStopping(monitor='loss', patience=2, verbose=2, mode='auto')
g_model.fit(train_x,train_y, epochs=25, batch_size=128,validation_split=0.2,verbose=2,callbacks=[earlyStopping])


# In[6]:


def distribution_index(fake):
    fake_idx=[]
    for i in fake:
    	w=np.argmax(i)
    	fake_idx.append([w])
    fake_idx=np.array(fake_idx)
    return fake_idx


def train_d(g_model,train_x,train_y):
    #partial=random.randint(4,12)

    train_x=train_x[check_stop]
    train_y=train_y[check_stop]
    earlyStopping=EarlyStopping(monitor='loss', patience=1, verbose=2, mode='auto')
    fake=g_model.predict(train_x)
    n = len(train_x)
    YT = np.zeros([n*2,2])
    YT[0:n,1] = 1
    YT[n:,0] = 1
    fake=distribution_index(fake)
    XT=np.concatenate([train_x,train_x])
    XT2=np.concatenate([train_y,fake])
    result=discriminator.fit([XT,XT2],YT, epochs=1,shuffle=True, batch_size=256, verbose=0,callbacks=[earlyStopping])
    return result.history['acc'][-1]

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances,paired_euclidean_distances,paired_cosine_distances

def distribution_index(fake):
    fake_idx=[]
    for i in fake:
    	w=np.argmax(i)
    	fake_idx.append([w])
    fake_idx=np.array(fake_idx)
    return fake_idx

def train_d(g_model,train_x,train_y):
    #partial=random.randint(4,12)
    train_x=train_x[check_stop]
    train_y=train_y[check_stop]
    earlyStopping=EarlyStopping(monitor='loss', patience=1, verbose=2, mode='auto')
    fake=g_model.predict(train_x)
    n = len(train_x)
    YT = np.zeros([n*2,2])
    YT[0:n,1] = 1
    YT[n:,0] = 1
    fake=distribution_index(fake)
    XT=np.concatenate([train_x,train_x])
    XT2=np.concatenate([train_y,fake])
    result=discriminator.fit([XT,XT2],YT, epochs=1,shuffle=True, batch_size=256, verbose=0,callbacks=[earlyStopping])
    return result.history['acc'][-1]
def finding_max_reward(train_x,train_y,count):
   #make_trainable(g_model,True)
    #attention_weight=attention_model.predict([train_x,train_y])[0]
    candidate=5
    new_fake=[]
    #sampling=num_words
    fake=g_model.predict(train_x)
    index=np.argsort(fake[0])
    index=index[::-1]
    #print(index)
    for i in index[0:candidate]:
            new_fake.append([i])
    fake_vector=[]
    input_x=train_x
    reward=[]
    vector=[]
    for j in new_fake:
        word=[]
        #next_sentence=[]
        index=g_model.predict(np.concatenate([train_x[:,1:que_pad],[j]],axis=1))
        index=np.argmax(index[0],axis=-1)
        #print(index)
        #next_sentence.append(index)      
        word.append(index)
        for i in range(count+1,4):
           train_x=np.concatenate([train_x[:,1:que_pad],[[index]]],axis=1)
           index=g_model.predict(train_x)
           index=np.argmax(index[0],axis=-1)
           #next_sentence.append(index) 
           word.append(index)
        vector.append(discriminator.predict([np.array(train_x),np.array([[index]])])[0][0])
        #reward.append(cosine_similarity([attention_weight],attention_model.predict([np.array(train_x),np.array([[index]])]))[0][0])
        #reward.append(cosine_similarity([attention_weight],attention_model.predict([np.array(train_x),np.array([[index]])]))[0][0])
        
    #print(reward)
    vector=vector-np.mean(vector)
    #reward=reward-np.mean(reward)
    #print(vector,reward)


    new_train=[] 
    for i in range(candidate):
             new_train.extend(input_x)
    #new_train=np.array(new_train)
    #new_fake=np.array(new_fake)
    #reward=np.array(reward)
    #reward=discriminator.predict([np.array(new_train),np.array(new_fake)])        
    #g_model.fit(new_train,new_fake,sample_weight=reward,batch_size=100,verbose=0)
    return new_train,new_fake,vector
def return_all_mean(train_x,train_y):
  fake=g_model.predict(train_x)
  #per=s_model.evaluate(train_x,train_y,batch_size=512)
  
  #per=1/per[0]
  mean=discriminator.predict([train_x,distribution_index(fake)])
  return np.mean(mean[:,1])



# In[7]:


def rollout(g_model,train_x,train_y):
    print('rollout')
    new_trainX=[]
    new_trainY=[]
    for key,i in enumerate(tqdm(train_x)):
        if key%20==0: 
            index=g_model.predict(np.array([i]))
            index=np.argmax(index[0],axis=-1)
            train_w=i
        else:
            train_w=np.concatenate([train_w[1:que_pad],[index]],axis=0) 
            index=g_model.predict(np.array([train_w]))
            index=np.argmax(index[0],axis=-1)  
        new_trainX.append(train_w)    
        new_trainY.append([index])
    new_trainX=np.array(new_trainX)
    new_trainY=np.array(new_trainY)
    return new_trainX,new_trainY
def train_d(g_model,train_x,train_y,X,Y):
    #train_x=train_x[::4]
    #train_y=train_y[::4]
    #X=X[::4]
    #Y=Y[::4]
    earlyStopping=EarlyStopping(monitor='loss', patience=1, verbose=2, mode='auto')
    fake=g_model.predict(train_x)
    n = len(train_x)
    YT = np.zeros([n*2,2])
    YT[0:n,1] = 1
    YT[n:,0] = 1
    #new_train=[]
    fake=distribution_index(fake)
    XT=np.concatenate([train_x,X])
    XT2=np.concatenate([train_y,Y])
    result=discriminator.fit([XT,XT2],YT, epochs=1,shuffle=True, batch_size=64, verbose=1,callbacks=[earlyStopping])
    return result.history['acc'][-1]


# In[8]:


def output_sequence(pair_train1,pair_train2,num):
    word2=[]
    test=[pair_train1[num][::-1]]
    test=np.array(test)
    index=g_model.predict(test)
    index=np.argmax(index[0],axis=-1)      
    word2.append(index)
    for i in range(1,que_pad):
       test=np.concatenate([test[:,1:que_pad],[[index]]],axis=1)
       index=g_model.predict(test)
       index=np.argmax(index[0],axis=-1)
       word2.append(index)
       if str(index) == str(word2idx['EOS']):
              break
    que=[]
    sample=[]
    test=[pair_train1[num]+pair_train2[num]]
    for g in test[0]:
          for value, age in word2idx.items():
                if age == g:
                	que.append(value)
    for g in word2:
          for value, age in word2idx.items():
                if age == g:
                	sample.append(value)
    print('question')
    print(''.join(que))
    print('RAP_model') 
    print(''.join(sample))
    que=que[0:20]+['   ans:   ']+que[ans_pad:]      
    return  ''.join(que),''.join(sample)
update_g=len(pair_train1)
for i in range(5):
    output_sequence(pair_train1,pair_train2,random.randint(0,random.randint(0,pair-1)))

old_result=0


# In[9]:


#from IPython.display import clear_output
epoch=0
start_time = time.time()
for x in range(10):
 #g_model.load_weights('partial_attention_half_attention.h5')   
    #old_policy=return_all_mean(train_x,train_y)
    print('d-step')
    X,Y=rollout(g_model,train_x,train_y)
    result=train_d(g_model,train_x,train_y,X,Y)
    del X,Y
    print(result)
    #if result == old_result:
    #    print('d-step-stop')
    #else:	
    #    old_result=result
    new_reward=0 
    #all_mean=return_all_mean(train_x,train_y[:update_g])
    #print('\n initial_now_all_reward: \n'+str(all_mean))
    print('finishsearch')
    count=0   
    dis_x=[]
    dis_y=[]
    reward=[]
    for g in tqdm(range(int(len(train_x)))):
     if g%20==0:
        code=0
     mode=mod_check_stop[g//20]
     for i in mode:
        if g%20<i:
            mod=i
            break
     if (mod-code)<0:      
         print(mod-code)
     s=finding_max_reward(train_x[g:g+1],train_y[g:g+1],(mod-code))
     dis_x.extend(s[0])
     dis_y.extend(s[1]) 
     reward.extend(s[2])
     code+=1   
     if g%1000==0:
        g_model.fit(np.array(dis_x),np.array(dis_y),epochs=0,batch_size=256,sample_weight=np.array(reward),class_weight='auto')
        dis_x=[]
        dis_y=[]
        reward=[]
     if g%10000==0:   
        for i in range(5):
            output_sequence(pair_train1,pair_train2,random.randint(0,random.randint(0,pair-1)))
    epoch=epoch+1
    fake=g_model.predict(train_x[:update_g])
    #mean=discriminator.predict([train_x[:update_g],distribution_index(fake)])
    #new_mean=np.mean(mean[:,1])
    #new_reward=return_all_mean(train_x[:update_g],train_y[:update_g])
    print('epoch',epoch) 
    print("--- %s seconds ---" % (time.time() - start_time))   
    #print('now_all_reward: '+str(new_reward))
 #g_model.save_weights('partial_attention_half_attention.h5')


# In[10]:


def output_sequence(test,model):
    test=test[:,::-1]
    word=[]
    next_sentence=[]
    index=model.predict(test)
    index=np.argmax(index[0],axis=-1)
    next_sentence.append(index)      
    word.append(index)
    for i in range(1,ans_pad):
        test=np.concatenate([test[:,1:que_pad],[[index]]],axis=1)
        index=model.predict(test)
        index=np.argmax(index[0],axis=-1)
        next_sentence.append(index) 
        word.append(index)
        if str(index) == str(word2idx['EOS']):
            break
    while len(next_sentence)<que_pad:
        next_sentence.append(str(word2idx['PAD']))
    ans=[]
    test=[pair_train1[num]+pair_train2[num]]
    for g in word:
          for value, age in word2idx.items():
                if age == g:
                	ans.append(value)
    #print('BattleGAN')                 
    #print(''.join(ans))
    return   np.array([next_sentence]), ''.join(ans),word  
f=open('2018/FULL_textrank_reward_parameter_6.csv', 'w', newline='',encoding = 'utf-8')

real=[]
sample=[]
for j in range(40):
    que=[]
    num=j*20#random.randint(0+j,random.randint(0,))
    test=[pair_train1[num]]
    test=np.array(test)
    for g in test[0]:
        for value, age in word2idx.items():
               if age == g:
                    	que.append(value)
    que=que[0:ans_pad]
    f.write(''.join(que)+'\n')                       
    for i in range(40):
        test=output_sequence(test,g_model)
        f.write(test[1]+'\n')
        test=test[0]
    f.write('\n')  
f.close()  

