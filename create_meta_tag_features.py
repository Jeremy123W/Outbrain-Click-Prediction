#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:53:39 2017

@author: jeremy
"""
import csv
import numpy as np
import datetime
import pandas as pd
start_time = datetime.datetime.now()
print("Beginning meta tag feature creation: ",start_time)

data_path = '../input/'


df = pd.read_csv(data_path+"documents_meta.csv",parse_dates=True)
df.publish_time=df.publish_time.astype(str)


df['date'],df['time']=df['publish_time'].str.split(' ',1).str
df['year'],df['month'],df['day']=df['date'].str.split('-',2).str
df['year_month']=df[['year','month']].astype(str).sum(axis=1)

df.to_csv(data_path+'documents_meta_times.csv',index=False)
print("Meta tag features created: ",datetime.datetime.now()-start_time)
