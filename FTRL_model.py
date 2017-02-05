#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:22:04 2016

@author: jeremy
"""


import csv,sys
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import pandas as pd


csv.field_size_limit(2**29)



##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
data_path = '../input/'
train = data_path+'clicks_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file
submission = 'sub_proba_interact.csv'  # path of to be outputted submission file

# B, model
alpha = .1  # learning rate
beta = 0   # smoothing parameter for adaptive learning rate
L1 = 10.    # L1 regularization, larger value means more regularized
L2 = 0     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 29             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions


# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = 87000000   # data after date N (exclusive) are used as validation
#holdafter= 5000
holdout = None # use every N training instance for holdout validation





##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path))):
        # process id
        disp_id = int(row['display_id'])
        ad_id = int(row['ad_id'])

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        x = []
        for key in row:
            if key!='display_id':
                x.append(abs(hash(key + '_' + row[key])) % D)

        row = prcont_dict.get(ad_id, [])		
        # build x 3values
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

        
            
        #ad_doc_id
        row = topcat_dict.get(ad_doc_id,[])
        x.append(abs(hash(topcat_header[0]+'_ad_id_'+str(row))) % D)

        row = top_topics_dict.get(ad_doc_id,[])
        x.append(abs(hash(top_topics_header[0]+'_ad_id_'+str(row))) % D)

        #row = top_entities_dict.get(ad_doc_id,[])
        #x.append(abs(hash(top_entities_header[0]+'_ad_id_'+str(row))) % D)     
        row = meta_times_dict.get(ad_doc_id,[])
        for ind, val in enumerate(row):
            x.append(abs(hash(meta_times_header[ind] + '_ad_id_' + str(val))) % D)
                
                
                
        row = event_dict.get(disp_id, [])
        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            #if ind!=6:
            x.append(abs(hash(event_header[ind] + '_' + val)) % D)
        
            
        #disp_doc_id    
        row = topcat_dict.get(disp_doc_id,[])
        x.append(abs(hash(topcat_header[0]+'_'+str(row))) % D)

        row = top_topics_dict.get(disp_doc_id,[])
        x.append(abs(hash(top_topics_header[0]+'_'+str(row))) % D)

        #row = top_entities_dict.get(disp_doc_id,[])
        #x.append(abs(hash(top_entities_header[0]+'_'+str(row))) % D)                
        row = meta_times_dict.get(disp_doc_id,[])
        for ind, val in enumerate(row):
            x.append(abs(hash(meta_times_header[ind] + '_disp_doc_id_' + str(val))) % D)
                
           
        #leak
        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):
            x.append(abs(hash('leakage_row_found_1'))%D)
        else:
            x.append(abs(hash('leakage_row_not_found'))%D)
            
        yield t, disp_id, ad_id, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

print("Content..")
with open(data_path + "promoted_content.csv") as infile:
	prcont = csv.reader(infile)
	#prcont_header = (prcont.next())[1:]
	prcont_header = next(prcont)[1:]
	prcont_dict = {}
	for ind,row in enumerate(prcont):
		prcont_dict[int(row[0])] = row[1:]
		if ind%100000 == 0:
			print(ind)
	print(len(prcont_dict))
del prcont

print("Events..")
with open(data_path + "events_with_time.csv") as infile:
	events = csv.reader(infile)
	#events.next()
	next(events)
	event_header = ['uuid', 'document_id', 'platform', 'geo_location','hour','day', 'loc_country', 'loc_state', 'loc_dma']
	event_dict = {}
	for ind,row in enumerate(events):
         tlist = row[1:3] + row[4:8]
         loc = row[5].split('>')
         if len(loc) == 3:
			tlist.extend(loc[:])
         elif len(loc) == 2:
			tlist.extend( loc[:]+[''])
         elif len(loc) == 1:
			tlist.extend( loc[:]+['',''])
         else:
			tlist.append(['','',''])
         event_dict[int(row[0])] = tlist[:] 
         if ind%100000 == 0:
			print("Events : ", ind)
	print(len(event_dict))
del events

###

print("top categories..")

with open("top_categories.csv") as infile:
	topcat = csv.reader(infile)
	topcat_header = next(topcat)[1]
	topcat_dict = {}
	for ind,row in enumerate(topcat):
		topcat_dict[int(row[0])] = row[1]
		if ind%100000 == 0:
			print(ind)
	print(len(topcat_dict))
del topcat 


print("top topics..")

with open("top_topics.csv") as infile:
	top_topics = csv.reader(infile)
	top_topics_header = next(top_topics)[1]
	top_topics_dict = {}
	for ind,row in enumerate(top_topics):
		top_topics_dict[int(row[0])] = row[1]
		if ind%100000 == 0:
			print(ind)
	print(len(top_topics_dict))
del top_topics 

with open("top_entities.csv") as infile:
	top_entities = csv.reader(infile)
	top_entities_header = next(top_entities)[1]
	top_entities_dict = {}
	for ind,row in enumerate(top_entities):
		top_entities_dict[int(row[0])] = row[1]
		if ind%100000 == 0:
			print(ind)
	print(len(top_entities_dict))
del top_entities 

with open(data_path+"documents_meta_times.csv") as infile:
     meta_times = csv.reader(infile)
     next(meta_times)
     meta_times_header =['source_id','publisher_id','date','year_month']
     meta_times_dict = {}
     for ind,row in enumerate(meta_times):
         meta_times_dict[int(row[0])] = [row[1],row[2],row[4],row[9]]
         if ind%100000 == 0:
             print(ind)
     print(len(meta_times_dict))
del meta_times

###

print("Leakage file..")
leak_uuid_dict= {}

with open(data_path+"leak_uuid_doc.csv") as infile:
	doc = csv.reader(infile)
	doc.next()
	leak_uuid_dict = {}
	for ind, row in enumerate(doc):
		doc_id = int(row[0])
		leak_uuid_dict[doc_id] = set(row[1].split(' '))
		if ind%100000==0:
			print("Leakage file : ", ind)
	print(len(leak_uuid_dict))
del doc
	

# start training
with open('cv_set.csv','w') as output:
    output.write('display_id,ad_id,clicked_prediction,clicked_actual\n')
    for e in range(epoch):
        loss = 0.
        count = 0
        date = 0
    
        for t, disp_id, ad_id, x, y in data(train, D):  # data is a generator
            #    t: just a instance counter
            # date: you know what this is
            #   ID: id provided in original data
            #    x: features
            #    y: label (click)
            #if t%2==0:
            # step 1, get prediction from learner
            p = learner.predict(x)

            if (holdafter and t > holdafter) or (holdout and t % holdout == 0):
                # step 2-1, calculate validation loss
                #           we do not train with the validation data so that our
                #           validation loss is an accurate estimation
                #
                # holdafter: train instances from day 1 to day N
                #            validate with instances from day N + 1 and after
                #
                # holdout: validate with every N instance, train with others
            
                
                output.write('%s,%s,%s,%s\n' % (disp_id, ad_id, str(p),str(y)))
                count += 1
            else:
                # step 2-2, update learner with label (click) information
                learner.update(x, p, y)

            if t%1000000 == 0:
                print("Processed : ", t, datetime.now())
            #if t == 1000000: #jrwatkin
             #  break
       



##########################################
##   calculate average precision
##########################################
if holdafter:
    df = pd.read_csv("cv_set.csv")
    df.ad_id = df.ad_id.astype(str)
    df2 = df.sort(['clicked_prediction'],ascending=False).groupby(['display_id'])['ad_id'].apply(' '.join).reset_index()
    df3=df[df.clicked_actual==1]
    del df3['clicked_prediction']
    del df3['clicked_actual']
    df3.columns=['display_id','ad_id_clicked']
    df4=pd.merge(df2,df3,on='display_id')
    
    a=df4['ad_id'][1].split()
    ap=0.

    for index,row in df4.iterrows():
        i=1
        for each in row['ad_id'].split():
            if each==row['ad_id_clicked']:
                ap+=1./i
            i+=1
    ap_tot=ap/len(df4) 
    print("Average precision = ",ap_tot) 

   
##############################################################################
# build predictions file file ################################################
##############################################################################

with open(submission, 'w') as outfile:
    outfile.write('display_id,ad_id,clicked\n')
    for t, disp_id, ad_id, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s,%s\n' % (disp_id, ad_id, str(p)))
        if t%1000000 == 0:
            print("Processed : ", t, datetime.now())

