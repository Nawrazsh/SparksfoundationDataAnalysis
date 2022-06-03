import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
data = pd.read_csv('student_scores.csv')
x = data[['Hours']]
y= data.Scores
plt.scatter(x, y)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
model =LinearRegression()
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25)
model.fit(xtrain.values,ytrain.values)
ypredic = model.predict(xtest.values)
print('%.3f'% mean_squared_error(ytest,ypredic))
print('%.3f'% r2_score(ytest,ypredic))
print(model.predict([[9.25]]))
