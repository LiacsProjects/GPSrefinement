
#from cProfile import label
#from turtle import color
#from joblib import PrintTime
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
#from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import MLPRegressor
#from geopy import distance 
#import tensorflow as tf
#from tensorflow import keras
#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import RegressorChain
# Import necessary modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
#from math import sqrt
#from sklearn.metrics import r2_score
#from datetime import timedelta
#from sklearn.model_selection import RepeatedKFold
#from keras import Sequential
#from keras.layers import Dense
#from numpy import absolute, mean
#from numpy import std
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR

#
#
#
merged = pd.read_csv('traintestset.csv')

merged.drop("time", axis=1, inplace=True)
#merged.drop("index", axis=1, inplace=True)

target_columns = ['uwbx','uwby']

predictors = list(set(list(merged.columns))-set(target_columns))

merged[predictors] = merged[predictors]/merged[predictors].max() # normalize value 

merged.describe().transpose()

X = merged[predictors].values
y = merged[target_columns].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=40)

model = LinearSVR()

# define the chained multioutput wrapper model
#wrapper = RegressorChain(model)
wrapper = MultiOutputRegressor(model)
# fit the model on the whole dataset
wrapper.fit(X_train, y_train)

results = wrapper.predict(X_test)
mea = mean_absolute_error(y_test,results)
mse = mean_squared_error(y_test,results)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test,results)

print("Mean absolute error:", mea)
print("Mean squared error:", mse)
print("Root mean squared error:",rmse)
print("Mean absolute percentage error:",mape*100)

xs =[]
ys =[]
xs1 =[]
ys1 =[]
xs2 =[]
ys2 =[]


for x in y_test:
    xs.append(x[0])
    ys.append(x[1])

for x in results:
    xs1.append(x[0])
    ys1.append(x[1])



baseline = pd.read_csv('traintestset.csv')

baseline.drop("time",   axis=1, inplace=True)
baseline.drop("b4lon",  axis=1, inplace=True)
baseline.drop("b4lat",  axis=1, inplace=True)
baseline.drop("b3dlon", axis=1, inplace=True)
baseline.drop("b3dlat", axis=1, inplace=True)
baseline.drop("b2dlon", axis=1, inplace=True)
baseline.drop("b2dlat", axis=1, inplace=True)
baseline.drop("b1dlon", axis=1, inplace=True)
baseline.drop("b1dlat", axis=1, inplace=True)
#baseline.drop("index", axis=1, inplace=True)



target_columns2 = ['uwbx','uwby'] 

predictors2 = list(set(list(baseline.columns))-set(target_columns2))

baseline[predictors2] = baseline[predictors2]/baseline[predictors2].max() # normalize value 

baseline.describe().transpose()

X = baseline[predictors2].values
y = baseline[target_columns2].values


X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.10, random_state=40)

model2 = LinearSVR()

# define the chained multioutput wrapper model
#wrapper = RegressorChain(model2)
wrapper = MultiOutputRegressor(model2)
# fit the model on the whole dataset
wrapper.fit(X_train2, y_train2)

results2 = wrapper.predict(X_test2)
mea = mean_absolute_error(y_test2,results2)
mse = mean_squared_error(y_test2,results2)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test2,results2)
print("Mean absolute error:",mea,"\n","Mean squared error:",mse,"\n","Root mean squared error:",rmse,"\n","Mean absolute percentage error:",mape*100)



for x in results2:
    xs2.append(x[0])
    ys2.append(x[1])
lat = list(merged.wlat)
lon = list(merged.wlon)

plt.figure() #figsize=(12,10))
plt.plot(xs,ys,label= 'UWB')
plt.plot(ys1,xs1, color = 'red',label='Predicted')
plt.plot(ys2,xs2, color = 'green',label='Baseline')


plt.legend()


plt.figure()
plt.plot(xs,ys,label= 'UWB')
plt.plot(lat,lon, color = 'black',label='GPS')
plt.legend()


plt.show()