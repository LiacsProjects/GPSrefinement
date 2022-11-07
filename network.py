# Import required libraries
from cProfile import label
from turtle import color
from joblib import PrintTime
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
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import RegressorChain
# Import necessary modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from datetime import timedelta
from sklearn.model_selection import RepeatedKFold
#from keras import Sequential
#from keras.layers import Dense
from numpy import mean
from numpy import std
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor


def gettime(traject,beacon1,beacon2,beacon3,beacon4,Toffset):
    Times0 = traject["seconds"] + traject["minute"]*60 + traject["hour"]*60*60 - Toffset
    Times1 = beacon1["seconds"] + beacon1["minute"]*60 + beacon1["hour"]*60*60 - Toffset
    Times2 = beacon2["seconds"] + beacon2["minute"]*60 + beacon2["hour"]*60*60 - Toffset
    Times3 = beacon3["seconds"] + beacon3["minute"]*60 + beacon3["hour"]*60*60 - Toffset
    Times4 = beacon4["seconds"] + beacon4["minute"]*60 + beacon4["hour"]*60*60 - Toffset
    return [Times0,Times1,Times2,Times3,Times4]

def maakdata(tr,time,Groundtruth1):  # traject tijdhuidig groundtruth
    closests = np.Infinity
    locx2 = []
    locy2 =[]
    a =  False
    start = int(time)
    for index in tr.index:
        huidig = start +  int(tr.time[index])
        Groundtruth1[(Groundtruth1['tijds'] > huidig)]
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

def vulbeaconcorrecties(beacon1,beacon2,beacon3,beacon4,beaconsplekken,timesbeacon):
    
    corr = pd.DataFrame(columns=['time','beacon','diflat','diflon','error']) # dataframe for all the corrections per beacon 
    
    for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon1.latitude[index]
        lon = beacon1.longitude[index]
        #beaconhere =  (beaconsplekken[0][0],beaconsplekken[0][1])        # beacon 1 locatie 
        punt = (lat,lon)
        #afstand = distance.great_circle(beaconhere,punt).m
        verschillat = beaconsplekken[0][0]-lat
        verschillon = beaconsplekken[0][1]-lon
        afstand = sqrt(verschillat*verschillat+verschillon*verschillon)
        time = beacon1.time[index] + timesbeacon[1]
        new_row = {'time': time , 'beacon':1, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

    
    for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon2.latitude[index]
        lon = beacon2.longitude[index] 
        #beaconhere =  (beaconsplekken[1][0],beaconsplekken[1][1])        # beacon 2 locatie 
        punt = (lat,lon)                                     # measured point 
        #afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beaconsplekken[1][0]-lat
        verschillon = beaconsplekken[1][1]-lon
        afstand = sqrt(verschillat*verschillat+verschillon*verschillon)        
        time = beacon2.time[index]+ timesbeacon[2]
        new_row = {'time': time , 'beacon':2, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

    
    for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon3.latitude[index]
        lon = beacon3.longitude[index] 
        #beaconhere =  (beaconsplekken[2][0],beaconsplekken[2][1])        # beacon 3 locatie 
        punt = (lat,lon)                                     # measured point 
        #afstand = distance.great_circle(beaconhere,punt).m
        verschillat = beaconsplekken[2][0]-lat
        verschillon = beaconsplekken[2][1]-lon
        afstand = sqrt(verschillat*verschillat+verschillon*verschillon)
        time = beacon3.time[index] + timesbeacon[3]
        new_row = {'time': time , 'beacon':3, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)

   
    for index in beacon4.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon4.latitude[index]
        lon = beacon4.longitude[index] 
        #beaconhere =  (beaconsplekken[3][0],beaconsplekken[3][1])        # beacon 4 locatie 
        punt = (lat,lon)                                     # measured point    
        #afstand = distance.great_circle(beaconhere,punt).m            
        verschillat = beaconsplekken[3][0]-lat
        verschillon = beaconsplekken[3][1]-lon
        afstand = sqrt(verschillat*verschillat+verschillon*verschillon)
        time = beacon4.time[index]+ timesbeacon[4]
        new_row = {'time': time , 'beacon':4, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        
    return corr

def mergecorrectiontraject(trajectframe,beaconframe,alltimes):
    frame = pd.DataFrame(columns=['beac1lat','beac1lon','beac2lat','beac2lon','beac3lat','beac3lon','beac4lat','beac4lon'])
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
        new_row = {'beac1lat': lats[0],'beac1lon': lons[0],'beac2lat': lats[1],'beac2lon': lons[1],'beac3lat': lats[2],'beac3lon':lons[2],'beac4lat':lats[3],'beac4lon':lons[3]}
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


#
# Lees data zonder header en beacon GPS positie
#
traject = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =",",skiprows=1) # Read trajectory
beacon11 = pd.read_csv('010 dae4 2022 03 16 12 12 56 gps.dat',sep =",",skiprows=1) 
beacon22 = pd.read_csv('020 db3a 2022 03 16 12 15 34 gps.dat',sep =",",skiprows=1) 
beacon33= pd.read_csv('030 dae4 2022 03 16 12 13 23 gps.dat',sep =",",skiprows=1) 
beacon44 = pd.read_csv('040 dac4 2022 03 16 12 13 36 gps.dat',sep =",",skiprows=1)

beaconsplekken=[(52.020161,4.268634),(52.020093,4.268767),(52.019981,4.268616),(52.020048,4.268484)]

columns1 = ["id","watch","year","month","day","hour","minute","seconds"]
traject2 = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =" ",nrows=1,names=columns1) 
bea1 = pd.read_csv('010 dae4 2022 03 16 12 12 56 gps.dat',sep =" ",nrows=1,names=columns1)
bea2 = pd.read_csv('020 db3a 2022 03 16 12 15 34 gps.dat',sep =" ",nrows=1,names=columns1) 
bea3 = pd.read_csv('030 dae4 2022 03 16 12 13 23 gps.dat',sep =" ",nrows=1,names=columns1) 
bea4 = pd.read_csv('040 dac4 2022 03 16 12 13 36 gps.dat',sep =" ",nrows=1,names=columns1)

#
# Ground truth
#
col_list = ["tagId","timestamp","dateTime","loc(x)","loc(y)"]
Groundtruth = pd.read_csv("test1.csv",usecols=col_list)
Groundtruth = Groundtruth[Groundtruth["tagId"] == 26694]  # filter out all, except tagid = 26694

print(Groundtruth)

Groundtruth["loc(x)"] = Groundtruth["loc(x)"].div(10000000) # to set the distance in meters instead of mm
Groundtruth["loc(y)"] = Groundtruth["loc(y)"].div(10000000)
Xuwb = Groundtruth["loc(x)"]
Yuwb = Groundtruth["loc(y)"]

print('Xuwb')
print(Xuwb)

#
# Plot the ground truth
#
plt.figure()
fig,ax = plt.subplots()
plt.plot(Xuwb, Yuwb, label= 'UWB')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("x [meter]")
plt.ylabel("y [meter]")
plt.legend()

Groundtruth.rename(columns = {'loc(x)':'locx', 'loc(y)':'locy'}, inplace = True)

#v = Groundtruth.dateTime.str.split() 
tijdseconden = []

# start time watch, this is used as reference time
Toffset = 45693

# Convert data time to seconds with as reference time = 0 being Toffset
for index in Groundtruth.index:
    seconds = get_seconds(Groundtruth.dateTime[index])
    tijdseconden.append(seconds-Toffset)

Groundtruth['tijdsec'] = tijdseconden

Groundtruth.drop("timestamp", axis=1, inplace=True)
Groundtruth.drop("dateTime", axis=1, inplace=True)
Groundtruth.drop("tagId", axis=1, inplace=True)

plt.figure()
fig,ax = plt.subplots()
plt.plot(tijdseconden,Xuwb,label= 'UWB x')
plt.plot(tijdseconden,Yuwb,label= 'UWB y')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("time [sec]")
plt.ylabel("x or y [meter/10000]")
plt.legend()

print("UWB")
print(Groundtruth)


# Calculate times in seconds in the day
tijden = gettime(traject2,bea1,bea2,bea3,bea4,Toffset)
print(tijden)

#tijdhuidig = tijden[0]
corrframe = vulbeaconcorrecties(beacon11,beacon22,beacon33,beacon44,beaconsplekken,tijden)

Xs1 = corrframe[corrframe['beacon'] == 1]['diflat']
Ys1 = corrframe[corrframe['beacon'] == 1]['diflon']
Xs2 = corrframe[corrframe['beacon'] == 2]['diflat']
Ys2 = corrframe[corrframe['beacon'] == 2]['diflon']
Xs3 = corrframe[corrframe['beacon'] == 3]['diflat']
Ys3 = corrframe[corrframe['beacon'] == 3]['diflon']
Xs4 = corrframe[corrframe['beacon'] == 4]['diflat']
Ys4 = corrframe[corrframe['beacon'] == 4]['diflon']

#
# Plot the ground truth
#
plt.figure()
fig,ax = plt.subplots()
plt.plot(Xs1,Ys1,label= 'Beacon1')
plt.plot(Xs2,Ys2,label= 'Beacon2')
plt.plot(Xs3,Ys3,label= 'Beacon3')
plt.plot(Xs4,Ys4,label= 'Beacon4')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("Difference latitude")
plt.ylabel("Difference longitude")
plt.legend()

Ts1 = corrframe[corrframe['beacon'] == 1]['time']
Es1 = corrframe[corrframe['beacon'] == 1]['error']
Ts2 = corrframe[corrframe['beacon'] == 2]['time']
Es2 = corrframe[corrframe['beacon'] == 2]['error']
Ts3 = corrframe[corrframe['beacon'] == 3]['time']
Es3 = corrframe[corrframe['beacon'] == 3]['error']
Ts4 = corrframe[corrframe['beacon'] == 4]['time']
Es4 = corrframe[corrframe['beacon'] == 4]['error']

plt.figure()
fig,ax = plt.subplots()
plt.plot(Ts1,Es1,label= 'Beacon1')
plt.plot(Ts2,Es2,label= 'Beacon2')
plt.plot(Ts3,Es3,label= 'Beacon3')
plt.plot(Ts4,Es4,label= 'Beacon4')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("Time")
plt.ylabel("Error = Sqrt(dif_lat^2 + dif_lon^2)")
plt.legend()

print("correctie data frame")
print(corrframe)

corrframe.drop('error',inplace=True, axis=1)

#
# Plot the traject of the watch
#

Ts = traject['time']
Xs = traject[' latitude'] - 52.020
Ys = traject[' longitude'] - 4.268

plt.figure()
fig,ax = plt.subplots()
plt.plot(Xs,Ys, label='Watch')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend()

plt.figure()
fig,ax = plt.subplots()
plt.plot(Ts,Xs, label='Watch lat')
plt.plot(Ts,Ys, label='Watch long')
plt.plot(tijdseconden, Xuwb, label='UWB x')
plt.plot(tijdseconden, Yuwb, label='UWB y')
#plt.xlim(0,12)
#plt.ylim(0,17)
plt.xlabel("time [sec]")
plt.ylabel("GPS coordinate")
plt.legend()

print("Traject")
print(traject)

#
# Make train set, time between 100 and 400, 300 seconds
#
#  X = (diflat1, diflon1, ..., diflat4, diflon4, trajectlat, trajectlon) in GPS coordinate unit
#  Y = (uwbx, uwby) in meters / 10,000
#



#
# Make test set, time between 500 and 800, 300 seconds
#
#  X = (diflat1, diflon1, ..., diflat4, diflon4, trajectlat, trajectlon) in GPS coordinate unit
#  Y = trained_model(X) = (predictedx, predictedy) in meters / 10,000, vergelijk met uwbx2 uwby2
#
#  X = (diflat1=0, diflon1=0, ..., diflat4=0, diflon4=0, trajectlat, trajectlon) in GPS coordinate unit
#  Y0 = trained_model(X) = (predictedx, predictedy) in meters / 10,000
#
#  Compare Y and Y0 where Y0 contains predictions without beacon error input.
#
traject = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =",",skiprows=1)
traject[' latitude'] = traject[' latitude'] - 52.020
traject[' longitude'] = traject[' longitude'] - 4.268

Wtime  = list(traject['time'])
Wlat   = list(traject[' latitude'])
Wlon   = list(traject[' longitude'])
Wtime_ = []
Wlat_  = []
Wlon_  = []

for i in range(0,len(Wtime)):
    if Wtime[i] >= 100.0 and Wtime[i] < 400.0:
        Wtime_.append(round(Wtime[i]))
        Wlat_.append(Wlat[i])
        Wlon_.append(Wlon[i])

print(len(Wtime))
print(len(Wtime_))

Groundtruth.rename(columns = {'tijdsec':'time'}, inplace = True)

'''
tomerge = []
for index in traject.index:
    huidig = tijden[0] + int(traject.time[index])
    filter = Groundtruth[(Groundtruth['tijdsec'] > float(huidig))]
    val = (filter['locx'].values[0],filter['locy'].values[0])
    tomerge.append(val)
df =  pd.DataFrame(tomerge,columns=['locx','locy'])
'''

#traject = traject.reset_index(drop=True)
result = pd.merge(traject, Groundtruth, on='time', how='inner')

print('traject')
print(traject)

print('groundtruth')
print(Groundtruth)

print('result')
print(result)

'''
corrframe['time']=corrframe['time'].astype('int')
merged = mergecorrectiontraject(result,corrframe,tijden)


gps = pd.DataFrame(merged[' latitude'])
gps2 = pd.DataFrame(merged[' longitude'])
gps3 = pd.DataFrame(columns=["latitude","longitude"])
gps3 = gps2.join(gps)


merged.drop(" private", inplace=True, axis=1)

# hier nog gps naar x en y invoegen  


target_columns = ['locx','locy'] 

predictors = list(set(list(merged.columns))-set(target_columns))

merged[predictors] = merged[predictors]/merged[predictors].max() # normalize value 

merged.describe().transpose()



X = merged[predictors].values

print("X features")
print(X)

y = merged[target_columns].values

print("Y output examples")
print(y)

#lengte = int(X.shape[0] * 0.6)
#X_train,X_test = np.array_split(X,2)
#y_train,y_test = np.array_split(y,2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=40)

print(X_train)
print(y_train)

# define base model
model = LinearSVR()
# define the chained multioutput wrapper model
wrapper = RegressorChain(model)
# fit the model on the whole dataset
wrapper.fit(X_train, y_train)

results = wrapper.predict(X_test)
mea = mean_absolute_error(y_test,results)
mse = mean_squared_error(y_test,results)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test,results)
print("Mean absolute error:",mea,"\n","Mean squared error:",mse,"\n","Root mean squared error:",rmse,"\n","Mean absolute percentage error:",mape*100)



gps3.rename(columns = {' longitude':'longitude', ' latitude':'latitude'}, inplace = True)
xgps = []
ygps=[]

for index in gps3.index:
    xgps.append(gps3.longitude[index])
    ygps.append(gps3.latitude[index])

gps3.to_csv("gps.csv")

xs =[]
ys =[]
xs1 =[]
ys1 =[]


for x in y_test:
    xs.append(x[0])
    ys.append(x[1])

for x in results:
    xs1.append(x[0])
    ys1.append(x[1])

plt.figure()
fig,ax = plt.subplots()
plt.plot(xs,ys,label= 'UWB')
plt.plot(xs1,ys1, color = 'red',label='predicted')
plt.xlim(0,12)
plt.ylim(0,17)
plt.xlabel("x [meter]")
plt.ylabel("y [meter]")
#img = plt.imread("veld3.png")
plt.legend()
#ax.imshow(img, extent=[0, 12, 0, 17])


plt.figure()
plt.plot(xgps,ygps, color = 'red',label='Raw GPS')
plt.ylim = (52.01998,52.02016)
plt.xlim = (4.26848,4.26876)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.show()
'''

# https://machinelearningmastery.com/multi-output-regression-models-with-python/


