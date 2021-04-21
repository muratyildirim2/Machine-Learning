# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:34:12 2021

@author: MURAT
"""
import pandas as pd 
import re
 

#import data
data = pd.read_csv("gender_classifier.csv",encoding = "latin1")

data = pd.concat([data.gender,data.description],axis=1)

data.dropna(axis=0,inplace=True) 

data.gender = [1 if each == "female" else 0 for each in data.gender]
#%%
import nltk

nltk.download("stopwords")


from nltk.corpus import stopwords

#%%


import nltk as nlp

description_list = []
for description in data.description : 
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = nlp.word_tokenize(description)
   # description = [ word for word in  description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    
#%% bag of words

from sklearn.feature_extraction.text import CountVectorizer
max_features= 4500

count_vect = CountVectorizer(max_features=max_features,stop_words="english")
space_matrix = count_vect.fit_transform(description_list).toarray()

print("En Cok Kullanılan {}  kelimeler {} ".format(max_features,count_vect.get_feature_names()))

# %%

y = data.iloc[:,0].values   
x = space_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# %% naive bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))


