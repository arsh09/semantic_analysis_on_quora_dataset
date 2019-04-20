
# -*- coding: utf-8 -*-
"""
The script trains the classifier on the sentence-vectors and their labels (similar or not)
The classifiers are SVM and KNN. They are saved in different files using pickle.

Arshad

"""

Debug = True
if Debug: print ("Importing the modules..")

import numpy as np
from sklearn import model_selection, neighbors,  svm , metrics
import  pickle
import pandas as pd

def train_classifier(feature_filename = "feature_set_file_without_questions_30_features.csv" , classifier_name = 'kNN'):
    
    dataframe = pd.read_csv(feature_filename)
    cols_to_drop = [i for i in range(1,31)]
    
    features = dataframe.drop(dataframe.columns[0],axis=1)
    labels = dataframe.drop(dataframe.columns[cols_to_drop],axis=1)
    
    
    # Convert the features and labels lists into numpy arrays 
    features = np.array((features),dtype=np.float64)
    labels = np.array((labels),dtype=np.float64).ravel()
    
    if Debug: print ("Shape of Feature Array: ", features.shape, "\nShape of Label Array: ", labels.shape)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features,labels,test_size = 0.05)
    
    if classifier_name.lower() == 'knn':
        classifier_name = 'kNN'
        if Debug: print ("Training the %s classifier" %(classifier_name))
        
        clf = neighbors.KNeighborsClassifier()     
        clf.fit(X_train, y_train) 
        
        print ((clf.predict(X_test[:5000]).shape,y_test[:5000].shape))
        
        if Debug: print ("Saving the %s classifier" %(classifier_name))
        
        with open(('%s_classifier_30_features.pkl' %classifier_name),'wb') as fid:
            pickle.dump(clf,fid)
        
        accuracy = 0
        for i in range (len(X_test)):
            if i%1000 == 0:
                print ("%d rows are done for prediction" %i)
            model_prediction =  clf.predict((X_test[i]).reshape(1,-1))
            if int(model_prediction[0]) == int(y_test[i]):
                accuracy = accuracy+1
                
            
        if Debug: print ("KNN Model is %d accurate " %(accuracy*100/len(y_test)))
        
        y_predict = clf.predict(X_test)
        recall_score = metrics.recall_score(y_test,y_predict)*100  # tp / (tp + fn)
        precision_score = metrics.precision_score(y_test,y_predict)*100  # tp / (tp + fp) 
        f1_score = metrics.f1_score(y_test,y_predict)*100 # F1 = 2 * (precision * recall) / (precision + recall)
        
        if Debug: print("Recall Score: %d percent\nPrecision Score: %d precent\nF1 Score: %d" %(recall_score,precision_score,f1_score))
    
    
    if classifier_name.lower() == 'svm':
        classifier_name = 'SVM'
        if Debug: print ("Training the %s classifier" %(classifier_name))
        
        clf = svm.SVC()     
        clf.fit(X_train, y_train) 
        
        print (clf.predict(X_test[:10]),y_test[:10])
        
        if Debug: print ("Saving the %s classifier" %(classifier_name))
        
        with open(('%s_classifier_30_features.pkl' %classifier_name),'wb') as fid:
            pickle.dump(clf,fid)
        
        accuracy = 0
        for i in range (len(X_test)):
            if i%1000 == 0:
                print ("%d rows are done for prediction" %i)
            model_prediction =  clf.predict((X_test[i]).reshape(1,-1))
            if int(model_prediction[0]) == int(y_test[i]):
                accuracy = accuracy+1
                
            
        if Debug: print ("SVM Model is %d accurate " %(accuracy*100/len(y_test)))
        
        y_predict = clf.predict(X_test)
        
        recall_score = metrics.recall_score(y_test,y_predict)*100  # tp / (tp + fn)
        precision_score = metrics.precision_score(y_test,y_predict)*100  # tp / (tp + fp) 
        f1_score = metrics.f1_score(y_test,y_predict)*100 # F1 = 2 * (precision * recall) / (precision + recall)
        
        if Debug: print("Recall Score: %d percent\nPrecision Score: %d precent\nF1 Score: %d" %(recall_score,precision_score,f1_score))
        
        
def main():
    train_classifier(classifier_name='knn')
    train_classifier(classifier_name='svm')
    
if __name__ == "__main__":
    main()
    
    
