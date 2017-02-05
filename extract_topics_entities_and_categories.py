#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:31:42 2016

@author: jeremy
"""

import datetime
import pandas as pd
start_time = datetime.datetime.now()
print(start_time)

data_path = '../input/'

df_new=pd.DataFrame()
df = pd.read_csv(data_path+"documents_topics.csv")
df2 = pd.read_csv(data_path+"documents_categories.csv")
df3 = pd.read_csv(data_path+"documents_entities.csv")



idx = df.groupby(['document_id'])['confidence_level'].transform(max) == df['confidence_level']
idx_rem = df.groupby(['document_id'])['confidence_level'].transform(max) != df['confidence_level']
dfout=df[idx]
df_rem=df[idx_rem]

#find second
idx_top2 = df_rem.groupby(['document_id'])['confidence_level'].transform(max) == df_rem['confidence_level']
#remove second
idx_rem2 = df_rem.groupby(['document_id'])['confidence_level'].transform(max) != df_rem['confidence_level']
#dataframe with all second highest
df_top2=df_rem[idx_top2]
#dataframe with first and second removed
df_rem2=df_rem[idx_rem2]

idx_top3 = df_rem2.groupby(['document_id'])['confidence_level'].transform(max) == df_rem2['confidence_level']
#remove second
#idx_rem3 = df_rem2.groupby(['document_id'])['confidence_level'].transform(max) != df_rem2['confidence_level']
#dataframe with all third highest
df_top3=df_rem2[idx_top3]



idx2 = df2.groupby(['document_id'])['confidence_level'].transform(max) == df2['confidence_level']
idx2_rem = df2.groupby(['document_id'])['confidence_level'].transform(max) != df2['confidence_level']
df2out=df2[idx2]
df2_rem=df2[idx2_rem]

#find second
idx2_2 = df2_rem.groupby(['document_id'])['confidence_level'].transform(max) == df2_rem['confidence_level']
#find all but second
idx2_rem2 = df2_rem.groupby(['document_id'])['confidence_level'].transform(max) != df2_rem['confidence_level']
#only second highest
df2_cat2=df2_rem[idx2_2]
#all but second highest
df2_rem2=df2_rem[idx2_rem2]

idx2_3 = df2_rem2.groupby(['document_id'])['confidence_level'].transform(max) == df2_rem2['confidence_level']
#third highest
df2_cat3=df2_rem2[idx2_3]


idx3 = df3.groupby(['document_id'])['confidence_level'].transform(max) == df3['confidence_level']
df3out=df3[idx3]



topics = df['topic_id'].unique()
print(len(df['document_id'].unique()))
print(df['confidence_level'].unique().mean())


print(datetime.datetime.now()-start_time)
print(datetime.datetime.now())
dfout.to_csv('top_topics.csv',index=False)
df2out.to_csv('top_categories.csv',index=False)
df3out.to_csv('top_entities.csv',index=False)
df_top2.to_csv('second_topics.csv',index=False)
df_top3.to_csv('third_topics.csv',index=False)
df2_cat2.to_csv('second_categories.csv',index=False)