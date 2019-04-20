# -*- coding: utf-8 -*-
"""
This script makes use of the trained kNN and SVM classifier
to predict if the two sentences are equivalent or not. The two questions
entered into the GUI.py, gets tokenized, the stop-words are removed, and each
word is converted into their respective vectors. Then the mean of each vector
is taken for a sentences. Hence a final 30-element long sentence vector is created
which then feeds into the trained kNN/SVM model to predict if the sentences are
similar of not.

Arshad
"""

Debug = False
if Debug: print ("Importing the modules")
from gensim.models import Word2Vec
import nltk 
import numpy as np
import pickle
    
def similarity_prediction(question_1 = "What is what",question_2= \
                          "where is what",number_of_feature = 30, \
                          word2vec_filename = 'model_binary_30_features.bin'\
                          ,clf_filename = "kNN_classifier_30_features.pkl"):    
    if Debug: print ("Loading the word2vec trained model named '%s'"\
                     %word2vec_filename)
    model = Word2Vec.load(word2vec_filename)
    
    
    if Debug: print ("Loading the trained classifer pickle named '%s'" \
                     %clf_filename)
    clf = pickle.load(open(clf_filename,'rb'))
    
    if Debug: print ("Tokenize the input questions and lowercase each word")
    
    question_1 = question_1.lower()
    question_2 = question_2.lower()
    
    token_1 = nltk.word_tokenize(question_1)
    token_2 = nltk.word_tokenize(question_2)
    
    new_feature =  np.zeros((number_of_feature,))
        
    for index_1 in range(len(token_1)):
        new_feature = new_feature + model[token_1[index_1]] 
        
    for index_2 in range(len(token_2)):
        new_feature = new_feature + model[token_2[index_2]] 
            
    new_feature = new_feature/(index_1+index_2)
    
    if Debug: print ("The size of feature vector for both inputted questions is:\
                     ",new_feature.shape)
    
    prediction = (clf.predict(new_feature.reshape(1,-1)))
    
    if int(prediction[0]) == 0:
        if Debug: print ("Questions are not similiar")
    else :
        if Debug: print ("Questions are similiar")
    
    return prediction

def main():    
    prediction = similarity_prediction(question_1="I am a boy",question_2="I am a man")

if __name__ == "__main__":
    main()
