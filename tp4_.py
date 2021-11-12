
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 07:31:09 2021
@author: Chaimae
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import  QFileDialog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import time

test=["A","A","B","B","C","C","D","D","E","E","F","F","G","H","H","I","I","J","J","K","L","M","M","N",
      "N","O","O","P","P","Q","R","S","S","T","U","U","V","W","X","Y","Z"]

da=[1,1,1,1,1,2,2,2,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,9,9,9,10,10,11,11,12,12,13,13,14,14,15,15,15,  #O
    16,16,17,17,18,18,19,19,20,20,21,21,21,22,22,23,23,24,24,25,25,26,26]
appD=["A","A","A","A","A","B","B","B","C","C","D","D","E","E","E",
    "F","F","G","G","H","H","I","I","I","I","I","J","J","K","K",
    "L","L","M","M","N","N","O","O","O","P", "P","Q","Q","R","R",
    "S", "S","T","T","U","U","U","V", "V","W","W","X","X","Y","Y","Z","Z"]
  
#pretraitement des images
data=[]
def alphabet_learning():    
    for i in range(1,63): 
        image=plt.imread("apprentissage/"+str(i)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
        img=np.array(image)                
        img=img.flatten()                
        data.append(img)        
    return data
   
def alphabet_test():
    for i in range(1,42): 
        image=plt.imread("TEST/"+str(i)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY) 
        img=np.array(image)
        img=img.flatten()                
        data.append(img) 
    return data 

images=alphabet_learning()
images_test=alphabet_test()
X_train,X_test,y_train,y_test=train_test_split(images,da,random_state=1,test_size=0.01)

#here Gaussian NB
gnb = GaussianNB()
s=time.time()
gnb.fit(images,appD)
stop=time.time()
# print("Le temps de l'apprentissage avec GaussianNB() : ",stop-s)
# print("Le taux de reconnaissance des images d'apprentissage est : ",gnb.score(images,appD) )
# print("occuracy GaussianNB on test ",gnb.score(images_test,test))

#here Multinomial NB
multiNB=MultinomialNB()
s=time.time()
multiNB.fit(images,appD)
stop=time.time()
# print("Le temps de l'apprentissage avec MultinomialNB() : ",stop-s)
# print("Le taux de reconnaissance des images d'apprentissage est : ",multiNB.score(images,appD) )
# print("here score occuracy Multinp for test ",multiNB.score(images_test,test))

#here Bernouilli NB
bern=BernoulliNB(alpha=0.005, binarize=0.41)
s=time.time()
bern.fit(images,appD)
stop=time.time()
#pred3=bern.predict(images)
# print("Le temps de l'apprentissage avec BernoulliNB() : ",stop-s)
# print("Le taux de reconnaissance des images est : ",bern.score(images,appD) )
# print("here score occuracy Bern for test ",bern.score(images_test,test))
image=plt.imread("TEST/2.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
img=np.array(image)                
vec=img.flatten()
pred=bern.predict([vec])
print(vec)
print("score test !!!!!  ",bern.score(vec,pred))
