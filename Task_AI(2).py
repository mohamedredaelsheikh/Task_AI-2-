# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:43:56 2021

@author: mohammed elsheikh
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
dataset=pd.read_csv("Wuzzuf_Jobs.csv")
dataset.describe()
dataset.sort_values("Title", inplace = True)
# dataset.drop_duplicates(subset ="Title",keep = "first", inplace = True)
numOfjops=dataset['Company'].value_counts().head(5)
print(numOfjops)
labels=["Confidential","EGIC","Mishkat Nour","Expand Cart","Majorel Egypt"]
plt.pie(numOfjops,labels = labels)
plt.show()
jops=dataset['Title'].value_counts().head(5)
print(jops)
plt.xlabel("jop_title")
plt.ylabel("NumberOfjop")
plt.title("most popular jop")
jopLabels=["Accountant","Sales","GraphicDesigner","Digital M","Sales Manager"]
plt.bar(jopLabels,jops)
plt.show()
areas=dataset['Location'].value_counts().head(5)
print(areas)
plt.xlabel("Location_Name")
plt.ylabel("NumberOfLocation")
plt.title("most popular location")
locLable=[" Cairo"," Maadi","New Cairo"," Nasr City","6th of October"]
plt.bar(locLable,areas)
plt.show()
print(dataset.Skills)
skills=dataset['Skills'].value_counts().head(5)
print(skills)