# -*- coding: utf-8 -*-
#this code compare handwrittig digit recognition predictions from logistic and multinomial regression 
#for better understanding of the tools utilized here please consult the website below
#http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/
#for further informations, read these recommended books
#Understanding Machine Learning: From Theory to Algorithms By: Shai Shalev-Shwartz and Shai Ben-David
#Mining of Massive Datasets By: Jure Leskovec, Anand Rajaraman, Jeff Ullman

"""
Created on Thu Mar 21 11:01:10 2019

@author: Mwaffo-Research
"""

#Reading the data
#load the necessary libraries
from scipy.io import loadmat
import os 
import pandas as pd

# import the necessary class
from sklearn.linear_model import LogisticRegression
# import the metrics class
from sklearn import metrics
from sklearn.metrics import classification_report

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#please download the dataset from this website link and save it on your hard drive
#https://s3.amazonaws.com/spark-public/ml/exercises/on-demand/machine-learning-ex3.zip
#read the data from your hard drive and change the directory if necessary using
os.getcwd()
#os.chdir('copy and paste file location here')
data = loadmat('ex3data1.mat')

#Selecting Feature and Target
X = data['X'] # Features here are digital values of hand writting pictures
y = data['y'] # Target variable are the corresponding integer values

#split dataset in features and target variable and training and test datasets
#70% train/30% test. Note you can change this to see how it affects the outcome
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)

#Visualizing sample of the data
_, axarr = plt.subplots(5,5,figsize=(5,5))
for i in range(5):
    for j in range(5):
        axarr[i,j].imshow(X_train[np.random.randint(X_train.shape[0])].\
        reshape((20,20), order = 'F'))
        axarr[i,j].axis('off')


#Visualizing Confusion Matrix using Heatmap
def plot_confus_mat(cnf_matrix):
    #matplotlib inline
    class_names=[0,1] # name of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
#Model Development and Prediction
def main():        
    #Building the logistic regression for multi-classification
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    # fit the model with data
    logreg.fit(X_train,y_train)
    
    #prediction
    y_pred=logreg.predict(X_test)
    
    #Model Evaluation using Confusion Matrix
    cnf_matrix_log = metrics.confusion_matrix(y_test, y_pred)
#    cnf_matrix_log 
    
    #Visualizing Confusion Matrix using Heatmap
    plot_confus_mat(cnf_matrix_log)
    
    #Confusion Matrix Evaluation Metrics
    print("Accuracy_logistic:",metrics.accuracy_score(y_test, y_pred))
    
    #Compute precision, recall, F-measure and support
    #for more explanation go to https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    print(classification_report(y_test, y_pred))
    
    
    #Implementing the multinomial logistic regression     
    # instantiate the model (using the default parameters)
    multireg = LogisticRegression(multi_class='multinomial', solver='newton-cg')#LogisticRegression()
    
    # Train multinomial logistic regression
    multireg.fit(X_train,np.ravel(y_train))
    
    #prediction
    y_pred=multireg.predict(X_test)
    
    #Model Evaluation using Confusion Matrix
    cnf_matrix_multi = metrics.confusion_matrix(y_test, y_pred)
    
    #Visualizing Confusion Matrix using Heatmap
    plot_confus_mat(cnf_matrix_multi)
    
    #Confusion Matrix Evaluation Metrics
    print("Accuracy_multinomial:",metrics.accuracy_score(y_test, y_pred))
    
    #Compute precision, recall, F-measure and support
    print(classification_report(y_test, y_pred))
    
if __name__ == "__main__":
    main()