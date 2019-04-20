# -*- coding: utf-8 -*-
"""
This script is the main script that runs a GUI for
ease of use for the prediction of semantic similarity.


Arshad
"""

import tkinter 
import Predict_Similarity as PredictSimilarity
 

def getquestions():
    
    if svm_text.get() == 1 and knn_text.get() == 0:
        print ("Running SVM Prediction")
        p = PredictSimilarity.similarity_prediction(question1_text.get(),question2_text.get(),clf_filename='SVM_classifier_30_features.pkl')

    elif svm_text.get() == 0 and knn_text.get() == 1:
        print ("Running SVM Prediction")
        p = PredictSimilarity.similarity_prediction(question1_text.get(),question2_text.get(),clf_filename='kNN_classifier_30_features.pkl')

    else:
        print ("Select a classifier")
        
        
    if int(p[0]) == 0:
        result_text.set("Similarity Result: Questions are not similiar")    
    else :
        result_text.set("Similarity Result: Questions are similiar")

def cleartext():
    question1_text.set("")
    question2_text.set("")
    result_text.set("Similarity Result: None")
    
    
def knn_select():
    if svm_text.get() == 1:
        svm_text.set(0)
        accuracy_val.set("Accuracy Score: 79%")
        recall_val.set("Recall Score: 67%")
        precision_val.set("Precision Score: 74%")
        f1_val.set("F1 Score: 71%")


def svm_select():
    if knn_text.get() == 1:
        knn_text.set(0)    
        accuracy_val.set("Accuracy Score: 76%")
        recall_val.set("Recall Score: 53%")
        precision_val.set("Precision Score: 74%")
        f1_val.set("F1 Score: 61%")

    

root = tkinter.Tk()
root.title('Similarity Project')
root.geometry("750x350")

# Variables to hold the questions texts
question1_text = tkinter.StringVar()
question2_text = tkinter.StringVar()
question1_text.set("Enter question # 01 here...")
question2_text.set("Enter question # 02 here...")
result_text = tkinter.StringVar()
result_text.set("Similarity Result: None")
result_text.set("")
svm_text = tkinter.IntVar()
svm_text.set(0)
knn_text = tkinter.IntVar()
knn_text.set(0)            

accuracy_val = tkinter.StringVar()
recall_val = tkinter.StringVar()
precision_val = tkinter.StringVar()
f1_val = tkinter.StringVar()
accuracy_val.set("Accuracy Score: None")
recall_val.set("Recall Score: None")
precision_val.set("Precision Score: None")
f1_val.set("F1 Score: None")

frame1 = tkinter.LabelFrame(root,text="Questions",bd=2)
frame1.grid(row=0,column=0,padx = 10,pady=10,sticky='nswe')

frame2 = tkinter.LabelFrame(root,text="Input",bd=2)
frame2.grid(row=1,column=0,padx=10,pady=10,sticky='nswe')

frame3 = tkinter.LabelFrame(root,text="Results",bd=2)
frame3.grid(row=2,column=0,padx=10,pady=10,sticky='nswe')

frame4 = tkinter.LabelFrame(frame2,text="Metrics",bd=2)
frame4.grid(row=0,column=2,padx=10,pady=10,sticky='nswe',rowspan=2)

accuracy_label = tkinter.Label(frame4,textvariable=accuracy_val)
accuracy_label.grid(row=0,column=0,sticky='w',padx=10,pady=2)
precision_label = tkinter.Label(frame4,textvariable=precision_val)
precision_label.grid(row=1,column=0,sticky='w',padx=10,pady=2)
recall_label = tkinter.Label(frame4,textvariable=recall_val)
recall_label.grid(row=0,column=1,sticky='w',padx=10,pady=2)
f1_label = tkinter.Label(frame4,textvariable=f1_val)
f1_label.grid(row=1,column=1,sticky='w',padx=10,pady=2)


question1_label = tkinter.Label(frame1,text="Question # 01")
question1_label.grid(row=0,column=0,padx = 10,pady=10,sticky='nswe')              
question1_entry = tkinter.Entry(frame1,text="Enter Question 1 here...",selectborderwidth=5,width=100,textvariable=question1_text)
question1_entry.grid(row=0,column=1,columnspan=3,padx = 10,pady=10,sticky='we')

question2_label = tkinter.Label(frame1,text="Question # 02")
question2_label.grid(row=1,column=0,padx = 10,pady=10,sticky='nswe')             
question2_entry = tkinter.Entry(frame1,text="Enter Question 2 here...",selectborderwidth=5,width=100,textvariable=question2_text)
question2_entry.grid(row=1,column=1,columnspan=3,padx = 10,pady=10,sticky='nswe')

submit_button = tkinter.Button(frame2,text="Get Prediction",width=30,command = getquestions).grid(row=0,column=0,padx=10,pady=10,sticky='we')
clear_button = tkinter.Button(frame2,text="Clear Text",width=30,command=cleartext).grid(row=1,column=0,padx=10,pady=10,sticky='we')

svm_label = tkinter.Checkbutton(frame2,text="Support Vector Machine",variable=svm_text,command=svm_select).grid(row=0,column=1,padx=10,pady=10,sticky='w')
knn_label = tkinter.Checkbutton(frame2,text="k-Nearest Neighbors",variable=knn_text,command=knn_select).grid(row=1,column=1,padx=10,pady=10,sticky='w')



result_label = tkinter.Label(frame3,textvariable=result_text).grid(row=0,column=0,padx=10,pady=10,sticky='we')

root.mainloop()
