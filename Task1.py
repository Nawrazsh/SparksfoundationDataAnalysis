import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('student_scores.csv')
x = data.Hours
y= data.Scores
xtrain,xtest,ytrain,ytest = train_test_split(x,y,.2)


