import pandas as pd
import numpy as np

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


data = pd.read_csv("SPAM text message 20170820 - Data.csv")

#Balancing the data set
ham   = data.loc[data.Category == 'ham']
spam  = data.loc[data.Category == 'spam']
spam2 = data.loc[data.Category == 'spam']
spam3 = data.loc[data.Category == 'spam']
spam4 = data.loc[data.Category == 'spam']
spam5 = data.loc[data.Category == 'spam']
spam6 = data.loc[data.Category == 'spam']
spam7 = data.loc[data.Category == 'spam'][:162]

df = pd.concat([ham, spam, spam2, spam3, spam4, spam5, spam6, spam7]) #Concat the data set
df = df.sample(frac=1).reset_index(drop = True)
df.Category.value_counts()

#Text Cleaning
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

import re
df['Message'] = df['Message'].apply(lambda x: re.sub(r'[^\w ]+','', x ))

#Removing multiple spaces
df['Message'] = df['Message'].apply(lambda x: ' '.join(x.split()))

#Stopwords removal
df['no_stop_w'] = df['Message'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

X = df['no_stop_w']
y = df['Category']


message_clf = Pipeline([
                    ('vect', TfidfVectorizer()),
                    ('clf', SVC(probability = True)),
                    ])

message_clf.fit(X, y)
