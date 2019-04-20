# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:34:51 2017

@author: Arshad
"""

from gensim.models import Word2Vec
import nltk 
import numpy as np


Debug = True

# load the data and save each question into a list. 
def load_data(quora_filename = "quora_duplicate_questions.tsv"):
    
    # import the question file
    question_file = open(quora_filename,encoding='utf8')
    
    # read the contents of the file
    question_data = question_file.read()
    
    if Debug: print ("File %s has been read" %quora_filename)
    
    # Read everyline except first (It has question ID headers)
    question_data = question_data.split("\n")[1:]
    
    if Debug: print ("Number of lines in data are: %d" %len(question_data))
    
    # List to hold question 1 and 2
    question_1 = []
    question_2 = []
    
    for i in (question_data):
        # Separate each line by tab
        tab_separated_line = i.split("\t")
        
        # if each line has 6 list elements then it means the 4th 
        # and 5th are the question
        if 6 == (len(tab_separated_line)):
            question_1.append(tab_separated_line[3])
            question_2.append(tab_separated_line[4])
    
    if Debug: print ("Number of question 1 is %d and number of question 2 is\
                     %d" %(len(question_1),len(question_2)))

    return question_1, question_2

# Preprocess the data. Like parsing for tab-delimated and get question 1 and 2 
def preprocess_data(question_1,question_2):
    
    tokens_question_1 = []
    tokens_question_2 = []
    
    for index in range(len(question_1)):
        i =  str(question_1[index]).strip("[").strip("]").strip("'")
        j =  str(question_2[index]).strip("[").strip("]").strip("'")
        i = i.lower()
        j = j.lower()
        tokens_question_1.append(nltk.word_tokenize(i))
        tokens_question_2.append(nltk.word_tokenize(j))
     
    print ("Length of both quesitions are: ",len(tokens_question_1)\
           ,len(tokens_question_2))
    
    sentences = tokens_question_1 + tokens_question_2
    return sentences

# I have already trained it once and saved it in a file
def train_model(sentences):
    # train model
    model = Word2Vec(sentences, min_count=1,workers=8,size=30)
    # summarize the loaded model
    print(model)
    # save model 
    model.wv.save_word2vec_format('model_text.txt', binary=False)
    model.save('model_binary_30_features.bin')
    
# Get features for each questions, take a mean of them and then save them in a file    
def get_features(model_filename='model_binary_30_features.bin'\
                 ,quora_filename="quora_duplicate_questions.tsv"\
                 ,number_of_features=30):
    
     # load model
    model = Word2Vec.load()
    get_features(model)

    # import the question file
    question_file = open(quora_filename,encoding='utf8')
    # read the contents of the file
    question_data = question_file.read()
    if Debug: print ("File %s has been read" %quora_filename)
    question_data = question_data.split("\n")[1:]
    if Debug: print ("Number of lines in data are: %d" %len(question_data))
    
    # List to hold question 1 and 2
    question_1 = []
    question_2 = []
    label = []
    features = []
    for i in (question_data):
        # Separate each line by tab
        tab_separated_line = i.split("\t")
        
        # if each line has 6 list elements then it means the 4th and 5th are the question
        if 6 == (len(tab_separated_line)):
            question_1.append(tab_separated_line[3])
            question_2.append(tab_separated_line[4])
            label.append(tab_separated_line[5])
       
    raw_feature = np.zeros((number_of_features,))
    
    for index in range(len(question_1)):
        i =  str(question_1[index]).strip("[").strip("]").strip("'")
        j =  str(question_2[index]).strip("[").strip("]").strip("'")
        i = i.lower()
        j = j.lower()
        
        raw_feature = np.zeros((number_of_features,))
        
        # Sum of each word feature (100) from each questions_1
        for count_i, w in enumerate((nltk.word_tokenize(i))):
            raw_feature = raw_feature + model[w]
         
        # Sum of each word feature (100) from each questions_2
        for count_j, w in enumerate((nltk.word_tokenize(j))):
            raw_feature = raw_feature + model[w]
          
        if (count_i + count_j) == 0:
            print (count_i,count_j,index)
        features.append(raw_feature/(count_i+count_j)) 
        
    if Debug: print ("Length of question set 1 is %d \n\
                     Length of question set 2 is %d \n\
                     Length of feature set is %d \n \
                     Length of label set is %d" \
                     %(len(question_1),len(question_2)\
                       ,len(features),len(label)))

    feature_file = open("feature_set_file_without\
                        _questions_30_features.csv","wb")    
    
    for index in range(len(question_1)):
        features_to_write = list(features[index])
        features_to_write = str(features_to_write)
        features_to_write = features_to_write.strip("[")
        features_to_write = features_to_write.strip("]")
        
#        data_to_write = str(question_1[index]) + "|" + str(question_2[index]) + "|" + str(label[index]) + "|" + str(list(features[index])).strip(["[","]"]) + "\n"
        data_to_write = str(label[index]) + "," + features_to_write + "\n"

        #   print (data_to_write)
        feature_file.write(data_to_write.encode('utf-8'))

    if Debug: print ("Feature file has been written successfully.")
    feature_file.close()
    
    

def main():
    # Sample Run of EACH FUNCTION
    q1, q2 = load_data()
    s = preprocess_data(q1,q2)
    train_model(s)

if __name__ == "__main__":
    main()
    
    