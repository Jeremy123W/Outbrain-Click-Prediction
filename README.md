# Outbrain-Click-Prediction

Top 7% Rank 68th/991 Solution
https://www.kaggle.com/jeremy123w

Outbrain as an online advertising company that needed help improving their recommendation algorithms to improve their
click-through rate.  The training and test sets were rather large being over 87 million and 32 million instances, respectively.
I used the FTRL-Proximal (follow the regularized leader) algorithm was used in my solution as described here: 

https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf

I implemented 27 different features for each training and test instance that I created by joining information from several 
separate csv files and doing some feature engineering.  I one hot encoded these features in my learning algorithm using a 
hashing trick.  I also used an online update approach to dramatically decrease the amount of RAM needed so the model could 
be trained on my machine.

The data sets can be downloaded here:
https://www.kaggle.com/c/outbrain-click-prediction/data

create_event_with_timestamp.py - creates time base features out of the timestamps

create_meta_tag_features.py - changes meta tags into useful features

extract_topics_entities_and_categories.py - creates feature lists for topics and categories by rank and also a feature list for entities

FTRL_model.py-processes all of the training data and trains an online model that creates predictions on the test set

create_submission.py takes all the predictions from the FTRL model and turns it into a submission file that can be evaluated by Kaggle
