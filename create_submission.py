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



df2=pd.DataFrame()
df = pd.read_csv("sub_proba_interact.csv")
df.ad_id = df.ad_id.astype(str)
df2 = df.sort(['clicked'],ascending=False).groupby(['display_id'])['ad_id'].apply(' '.join).reset_index()

df2.to_csv('submission_file.csv',index=False)
print(datetime.datetime.now()-start_time)
