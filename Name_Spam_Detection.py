    # -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Kanna Chandrasekaran
Alias: kach@microsoft.com
Date: 10/17/2018
Description: Python script to pre-populate Experience | Entity | sub-entity

Ref:https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
"""

import pandas as pd

#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer

def split_grams(in_str,l=3):
    out_str=''
    in_str=str(in_str).lower()
    in_str=in_str.lstrip().rstrip()
    in_str=' '+in_str
    for i in range(0,len(in_str)-l+1):
        out_str=out_str+' '+in_str[i:i+l]
    return out_str


if __name__=="__main__":    
#------------------------------------------------------------------------------    
# Syntax coloring
    W  = '\033[0m'  # white (normal)
    R  = '\033[1;31m' # red
    G  = '\033[1;32m' # green
    O  = '\033[1;33m' # orange
    B  = '\033[34m' # blue
    P  = '\033[35m' # purple
    
#    TRAINING
# First name
    df_trainFN=pd.read_excel('Train_Firstname.xlsx')
    df_trainFN['grams']=df_trainFN['Firstname'].apply(split_grams)
            
    x_train=df_trainFN['grams']    
    y_train=df_trainFN['Isvalid']
    
    count_vect=CountVectorizer()
    x_train_counts=count_vect.fit_transform(x_train)
        
    tfidf_transformer=TfidfTransformer()
    x_train_tfidf=tfidf_transformer.fit_transform(x_train_counts)
    
    clf=LinearSVC().fit(x_train_tfidf,y_train)

## Last name
#    df_trainLN=pd.read_excel('Train_Lastname.xlsx')
#    df_trainLN['grams']=df_trainLN['Lastname'].apply(split_grams)
#            
#    x1_train=df_trainLN['grams']    
#    y1_train=df_trainLN['Isvalid']
#    
#    count_vect1=CountVectorizer()
#    x1_train_counts=count_vect1.fit_transform(x1_train)
#        
#    tfidf_transformer1=TfidfTransformer()
#    x1_train_tfidf=tfidf_transformer1.fit_transform(x1_train_counts)
#    
#    clf1=LinearSVC().fit(x1_train_tfidf,y1_train)

    while True:
        FN=input('> First Name:')
        LN=input('> Last Name:')
        
        if FN=='~' and LN=='~':
            break
        
        print('')
        
        if clf.predict(count_vect.transform([split_grams(FN)]))[0]=='INVALID':
            FN1=R+FN+W
        else:
            FN1=G+FN+W

        if clf.predict(count_vect.transform([split_grams(LN)]))[0]=='INVALID':
            LN1=R+LN+W
        else:
            LN1=G+LN+W          
        
        
#        print(LN,clf1.predict(count_vect1.transform([split_grams(LN)])))
        print('\t',FN1,LN1)
        print('')
        print('\t',FN,clf.predict(count_vect.transform([split_grams(FN)])),LN,clf.predict(count_vect.transform([split_grams(LN)])))
        print('\n--------------------------------')


#------------------------------------------------------------------------------
## In[]
## Load libraries
#import pandas as pd
#from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
#from sklearn.model_selection import train_test_split # Import train_test_split function
#from sklearn import metrics
#
### TESTING OTHER CLASSIFIERS
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import svm
#
##Reading Training Data
#X=df_trainFN[['grams']]
#
#count_vect=CountVectorizer()
#x_train_counts=count_vect.fit_transform(x_train)
#tfidf_transformer=TfidfTransformer()
#x_train_tfidf=tfidf_transformer.fit_transform(x_train_counts)
#X=x_train_tfidf
#
#y=df_trainFN.iloc[:,-1]
#
## Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
#
#
#clf = DecisionTreeClassifier()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("DecisionTree:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'DecisionTreeClassifier']=metrics.accuracy_score(y_test, y_pred)
#
#clf = MultinomialNB()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("MultinomialNB:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'MultinomialNB']=metrics.accuracy_score(y_test, y_pred)
#
#clf = GaussianNB()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("GaussianNB:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'GaussianNB']=metrics.accuracy_score(y_test, y_pred)
#
#clf = LogisticRegression()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("LogisticRegression:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'LogisticRegression']=metrics.accuracy_score(y_test, y_pred)
#
#clf = SGDClassifier()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("SGDClassifier:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'SGDClassifier']=metrics.accuracy_score(y_test, y_pred)
#
#clf = KNeighborsClassifier()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("KNeighborsClassifier:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'KNeighborsClassifier']=metrics.accuracy_score(y_test, y_pred)
#
#clf = GradientBoostingClassifier()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("GradienBoostingClassifier:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'GradientBoostingClassifier']=metrics.accuracy_score(y_test, y_pred)
#
#clf = svm.SVC()
#clf = clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("svm:",metrics.accuracy_score(y_test, y_pred))
##df_compare.loc[Test,'svm']=metrics.accuracy_score(y_test, y_pred)
#
##df_compare.to_excel('df_MLmodels_Compare.xlsx')
