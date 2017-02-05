#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 13:55:54 2017

@author: jeremy
"""

import numpy as np
import datetime
import pandas as pd
start_time = datetime.datetime.now()
print(start_time)

data_path = '../input/'


df = pd.read_csv(data_path+"events.csv")
df['hour']=(df.timestamp/(3600*1000) % 24)
df['day']=(df.timestamp/(3600*24*1000))
df['hour'] =df.hour.round(0)

df['loc_country'],df['loc_state'],df['loc_dma']= df['geo_location'].str.split('>',2).str

df.day = df.day.astype(int)
df.to_csv(data_path+'events_with_time.csv',index=False)
print("Events with hour and day created: ",datetime.datetime.now()-start_time)