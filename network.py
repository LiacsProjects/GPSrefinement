# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from geopy import distance 
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
# Import necessary modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from datetime import timedelta
from sklearn.model_selection import RepeatedKFold
from keras import Sequential
from keras.layers import Dense
from numpy import mean
from numpy import std
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

def gettime(ttraject,beacon1,beacon2,beacon3,beacon4,beacon5):
    Times0 = ttraject["seconds"] + ttraject["minute"]*60 + ttraject["hour"]*60*60
    Times1 = beacon1["seconds"] + beacon1["minute"]*60 + beacon1["hour"]*60*60
    Times2 = beacon2["seconds"] + beacon2["minute"]*60 + beacon2["hour"]*60*60
    Times3 = beacon3["seconds"] + beacon3["minute"]*60 + beacon3["hour"]*60*60
    Times4 = beacon4["seconds"] + beacon4["minute"]*60 + beacon4["hour"]*60*60
    Times5 = beacon5["seconds"] + beacon5["minute"]*60 + beacon5["hour"]*60*60
    tt = [Times0,Times1,Times2,Times3,Times4,Times5]
    return tt


def maakdata(tr,time,Groundtruth1):
    closests = np.Infinity
    locx2 = []
    locy2 =[]
    a =  False
    for index in tr.index:
        time = int(time) +  int(tr.time[index])
        Groundtruth1[(Groundtruth1['tijds'] > time)]
        timegroundtruth = Groundtruth1.tijds[0]   
        locatiex = Groundtruth1.locx[0]
        locatiey = Groundtruth1.locy[0]
        locx2.append((locatiex))
        locy2.append((locatiey))
    tr['locx'] = locx2
    tr['locy'] = locy2
    return tr

def get_seconds(time_str):
    hh, mm, ss = time_str.split(':')
    sss,ns = ss.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(sss)

def vulbeaconcorrecties(beacon1,beacon2,beacon3,beacon4,beacon5,beacons1,timesbeacon):
    corr = pd.DataFrame(columns=['time','beacon','diflat','diflon','error']) # dataframe for all the corrections per beacon 
    
    for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon1.latitude[index]
        lon = beacon1.longitude[index]
        beaconhere =  (beacons1[0][0],beacons1[0][1])        # beacon 1 locatie 
        punt = (lat,lon)
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[0][0]-lat
        verschillon = beacons1[0][1]-lon
        time = beacon1.time[index] + timesbeacon[1]
        new_row = {'time': time , 'beacon':1, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

    
    for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon2.latitude[index]
        lon = beacon2.longitude[index] 
        beaconhere =  (beacons1[1][0],beacons1[1][1])        # beacon 2 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[1][0]-lat
        verschillon = beacons1[1][1]-lon
        time = beacon2.time[index]+ timesbeacon[2]
        new_row = {'time': time , 'beacon':2, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

    
    for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon3.latitude[index]
        lon = beacon3.longitude[index] 
        beaconhere =  (beacons1[2][0],beacons1[2][1])        # beacon 0 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[2][0]-lat
        verschillon = beacons1[2][1]-lon
        time = beacon3.time[index] + timesbeacon[3]
        new_row = {'time': time , 'beacon':3, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

   
    for index in beacon4.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon4.latitude[index]
        lon = beacon4.longitude[index] 
        beaconhere =  (beacons1[3][0],beacons1[3][1])        # beacon 0 locatie 
        punt = (lat,lon)                                     # measured point    
        afstand = distance.great_circle(beaconhere,punt).m            
        verschillat = beacons1[3][0]-lat
        verschillon = beacons1[3][1]-lon
        time = beacon4.time[index]+ timesbeacon[4]
        new_row = {'time': time , 'beacon':4, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

    
    
    for index in beacon5.index  :      # een eventuele 5 want dan kan er gekeken worden naar als alle beacons op dezelfde locatie staan
        lat = beacon5.latitude[index]
        lon = beacon5.longitude[index] 
        beaconhere =  (beacons1[4][0],beacons1[4][1])        # beacon 0 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m            
        time = beacon5.time[index]+ timesbeacon[5]
        verschillat = beacons1[4][0] - lat 
        verschillon = beacons1[4][1] - lon 
        new_row = {'time': time , 'beacon':5, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        
    return corr

def mergecorrectiontraject(trajectframe,beaconframe,alltimes):
    frame = pd.DataFrame(columns=['beac1lat','beac1lon','beac2lat','beac2lon','beac3lat','beac3lon','beac4lat','beac4lon','beac5lat','beac5lon'])
    lats = []
    lons=[]
    temp = beaconframe
    for index in trajectframe.index:
        huidigetijd = trajectframe.time[index]+alltimes[0]
        for  i in range(1,6):
            tijdverschil = np.Infinity
            beaconframe = temp 
            beaconframe[(beaconframe['beacon'] == i)]
            beaconframe[(beaconframe['time']  >= int(huidigetijd)) ]
            lats.append(beaconframe.diflat[0])
            lons.append(beaconframe.diflon[0])
        new_row = {'beac1lat': lats[0],'beac1lon': lons[0],'beac2lat': lats[1],'beac2lon': lons[1],'beac3lat': lats[2],'beac3lon':lons[2],'beac4lat':lats[3],'beac4lon':lons[3],'beac5lat':lats[4],'beac5lon':lons[4]}
        frame = frame.append(new_row,ignore_index = True)
    merged = pd.concat([trajectframe,frame],axis=1,join='inner')
    return merged

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model
 
# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results

# gebruiken om hele bestanden te lezen 
traject = pd.read_csv('001 dc81 2022 01 14 11 59 22 gps.dat',sep =",",skiprows=1) # Read trajectory
beacon11 = pd.read_csv('011 dc81 2022 01 14 11 59 27 gps.dat',sep =",",skiprows=1) 
beacon22 = pd.read_csv('021 dc81 2022 01 14 11 59 54 gps.dat',sep =",",skiprows=1) 
beacon33= pd.read_csv('031 dc81 2022 01 14 12 00 27 gps.dat',sep =",",skiprows=1) 
beacon44 = pd.read_csv('041 dc81 2022 01 14 12 00 45 gps.dat',sep =",",skiprows=1)    
beacon55 = pd.read_csv('051 dc81 2022 01 14 12 01 17 gps.dat',sep =",",skiprows=1) 
beaconsplekken=[(52.020093,4.268766),(52.019918,4.268743),(52.0199568,4.268911),(52.019814,4.268948),(52.0198887,4.2692081)]

# volgende gebruiken voor de eerste om eerste line eruit te halen 
columns1 = ["id","watch","year","month","day","hour","minute","seconds"]
traject2 = pd.read_csv('001 dc81 2022 01 14 11 59 22 gps.dat',sep =" ",nrows=1,names=columns1) # Read trajectory for the first line 
bea1 = pd.read_csv('011 dc81 2022 01 14 11 59 27 gps.dat',sep =" ",nrows=1,names=columns1)
bea2 = pd.read_csv('021 dc81 2022 01 14 11 59 54 gps.dat',sep =" ",nrows=1,names=columns1) 
bea3 = pd.read_csv('031 dc81 2022 01 14 12 00 27 gps.dat',sep =" ",nrows=1,names=columns1) 
bea4 = pd.read_csv('041 dc81 2022 01 14 12 00 45 gps.dat',sep =" ",nrows=1,names=columns1)    
bea5 = pd.read_csv('051 dc81 2022 01 14 12 01 17 gps.dat',sep =" ",nrows=1,names=columns1) 

col_list = ["timestamp","dateTime","loc(x)","loc(y)"]
Groundtruth = pd.read_csv("test1.csv",usecols=col_list)# ground truth opzoeken
Groundtruth["loc(x)"] = Groundtruth["loc(x)"].div(1000) # to set the distance in meters instead of mm
Groundtruth["loc(y)"] = Groundtruth["loc(y)"].div(1000)
Groundtruth.rename(columns = {'loc(x)':'locx', 'loc(y)':'locy'}, inplace = True)

v = Groundtruth.dateTime.str.split() 
tijdseconden = []

for index in Groundtruth.index:
    seconds = get_seconds(Groundtruth.dateTime[index])
    tijdseconden.append(seconds)

Groundtruth['tijds'] = tijdseconden


# preprocessen van de data en het maken van een compleet dataframe met de goede waarde voor een voorspelling 
tijden = gettime(traject2,bea1,bea2,bea3,bea4,bea5)
tijdhuidig = tijden[0]
corrframe = vulbeaconcorrecties(beacon11,beacon22,beacon33,beacon44,beacon55,beaconsplekken,tijden)
corrframe.drop('error',inplace=True, axis=1)
Fullframe = maakdata(traject,tijdhuidig,Groundtruth)
corrframe['time']=corrframe['time'].astype('int')
merged = mergecorrectiontraject(Fullframe,corrframe,tijden)
merged.drop('private', inplace=True, axis=1)

target_columns = ['locx','locy'] 
predictors = list(set(list(merged.columns))-set(target_columns))
merged[predictors] = merged[predictors]/merged[predictors].max()
merged.describe().transpose()

X = merged[predictors].values
y = merged[target_columns].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)



# evaluate model
#results = evaluate_model(X, y) kan gebruikt worden voor resultaten functie
# summarize performance
#print('MAE: %.3f (%.3f)' % (mean(results), std(results))) # kan gebruikt worden om data te verkrijgen van de analyse
#mlp.fit(X_train,y_train)


# mulkti output model invoegen 



# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables

# https://machinelearningmastery.com/multi-output-regression-models-with-python/

# define model
model = LinearSVR()
# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# fit the model on the whole dataset
wrapper.fit(X_train, y_train)
# make a single prediction

yhat = wrapper.predict(X_test)
# summarize the prediction
print('Predicted: %s' % yhat)
