# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 23:47:57 2018

@author: Dell
"""
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

#Importing Dataset
reviews=load_files('txt_sentoken/')
X,y=reviews.data,reviews.target

#Storing as pickle files
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
"""Unpickling dataset
with open('X.pickle','rb') as f:
    X=pickle.load(f)"""

#Pre-processing data and crearing corpus
corpus=[]
for i in range(0,len(X)):
    review=re.sub(r'\W',' ',str(X[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()

#Converting to tf-idf model
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()                                       

#using tfidf vectorizer directly(either use this or the above 2 sections..not both)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()


#Splitting into training and test
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(X,y,test_size=0.2,random_state=0)

                                                                               
#Training Classifier
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)

sent_pred=classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)

#Pickling the vectorizer so it can be used for later projects
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
#pickling tfidf vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#Unpickling the classifier and vectorizer
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)

sample=["You are a genius"]
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample))
