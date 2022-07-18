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
from keras import Sequential
from keras.layers import Dense
from numpy import absolute, mean
from numpy import std
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.svm import SVR

def gettime(ttraject,beacon1,beacon2,beacon3,beacon4):
    Times0 = ttraject["seconds"] + ttraject["minute"]*60 + ttraject["hour"]*60*60
    Times1 = beacon1["seconds"] + beacon1["minute"]*60 + beacon1["hour"]*60*60
    Times2 = beacon2["seconds"] + beacon2["minute"]*60 + beacon2["hour"]*60*60
    Times3 = beacon3["seconds"] + beacon3["minute"]*60 + beacon3["hour"]*60*60
    Times4 = beacon4["seconds"] + beacon4["minute"]*60 + beacon4["hour"]*60*60
    tt = [Times0,Times1,Times2,Times3,Times4]
    return tt


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

def vulbeaconcorrecties(beacon1,beacon2,beacon3,beacon4,beacons1,timesbeacon):
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
"""
# gebruiken om hele bestanden te lezen 
traject = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =",",skiprows=1) # Read trajectory
beacon11 = pd.read_csv('010 dae4 2022 03 16 12 12 56 gps.dat',sep =",",skiprows=1) 
beacon22 = pd.read_csv('020 db3a 2022 03 16 12 15 34 gps.dat',sep =",",skiprows=1) 
beacon33= pd.read_csv('030 dae4 2022 03 16 12 13 23 gps.dat',sep =",",skiprows=1) 
beacon44 = pd.read_csv('040 dac4 2022 03 16 12 13 36 gps.dat',sep =",",skiprows=1)    
beaconsplekken=[(52.020161,4.268634),(52.020093,4.268767),(52.019981,4.268616),(52.020048,4.268484)]

# volgende gebruiken voor de eerste om eerste line eruit te halen 
columns1 = ["id","watch","year","month","day","hour","minute","seconds"]
traject2 = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =" ",nrows=1,names=columns1) # Read trajectory for the first line 
bea1 = pd.read_csv('010 dae4 2022 03 16 12 12 56 gps.dat',sep =" ",nrows=1,names=columns1)
bea2 = pd.read_csv('020 db3a 2022 03 16 12 15 34 gps.dat',sep =" ",nrows=1,names=columns1) 
bea3 = pd.read_csv('030 dae4 2022 03 16 12 13 23 gps.dat',sep =" ",nrows=1,names=columns1) 
bea4 = pd.read_csv('040 dac4 2022 03 16 12 13 36 gps.dat',sep =" ",nrows=1,names=columns1)    




col_list = ["tagId","timestamp","dateTime","loc(x)","loc(y)"]
Groundtruth = pd.read_csv("test1.csv",usecols=col_list)# ground truth opzoeken
Groundtruth["loc(x)"] = Groundtruth["loc(x)"].div(1000) # to set the distance in meters instead of mm
Groundtruth["loc(y)"] = Groundtruth["loc(y)"].div(1000)
Xs = Groundtruth["loc(x)"]
Ys = Groundtruth["loc(y)"]

xys = pd.concat([Xs,Ys],axis=1,join='inner') # frame of locx and locy

Groundtruth.rename(columns = {'loc(x)':'locx', 'loc(y)':'locy'}, inplace = True)


v = Groundtruth.dateTime.str.split() 
tijdseconden = []



for index in Groundtruth.index:
    seconds = get_seconds(Groundtruth.dateTime[index])
    tijdseconden.append(seconds)

Groundtruth['tijds'] = tijdseconden


# preprocessen van de data en het maken van een compleet dataframe met de goede waarde voor een voorspelling 
tijden = gettime(traject2,bea1,bea2,bea3,bea4)
tijdhuidig = tijden[0]
corrframe = vulbeaconcorrecties(beacon11,beacon22,beacon33,beacon44,beaconsplekken,tijden)
corrframe.drop('error',inplace=True, axis=1)

Groundtruth.drop("timestamp", axis=1, inplace=True)
Groundtruth.drop("dateTime", axis=1, inplace=True)

Groundtruth = Groundtruth[Groundtruth["tagId"] == 26694]  # change the tagid here t
Groundtruth.drop("tagId", axis=1, inplace=True)


verschil = 45764 - tijden[0]
traject = traject[traject['time']<300]
traject = traject[traject['time'] >= int(verschil)]


tomerge = []
for index in traject.index:
    huidig =  tijden[0]+  int(traject.time[index])
    filter = Groundtruth[(Groundtruth['tijds'] > float(huidig))]
    val = (filter['locx'].values[0],filter['locy'].values[0])
    tomerge.append(val)
df =  pd.DataFrame(tomerge,columns=['locx','locy'])


traject = traject.reset_index(drop=True)
result = traject.join(df)


corrframe['time']=corrframe['time'].astype('int')
merged = mergecorrectiontraject(result,corrframe,tijden)


gps = pd.DataFrame(merged[' latitude'])
gps2 = pd.DataFrame(merged[' longitude'])
gps3 = pd.DataFrame(columns=["latitude","longitude"])
gps3 = gps2.join(gps)


merged.drop(" private", inplace=True, axis=1)

# hier nog gps naar x en y invoegen  
"""
merged = pd.read_csv('traintestset.csv')

merged.drop("time", axis=1, inplace=True)

merged.drop("index", axis=1, inplace=True)

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
print("Mean absolute error:",mea,"\n","Mean squared error:",mse,"\n","Root mean squared error:",rmse,"\n","Mean absolute percentage error:",mape*100)


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

baseline.drop("time", axis=1, inplace=True)
baseline.drop("b4lon", axis=1, inplace=True)
baseline.drop("b4lat", axis=1, inplace=True)
baseline.drop("b3dlon", axis=1, inplace=True)
baseline.drop("b3dlat", axis=1, inplace=True)
baseline.drop("b2dlon", axis=1, inplace=True)
baseline.drop("b2dlat", axis=1, inplace=True)
baseline.drop("b1dlon", axis=1, inplace=True)
baseline.drop("b1dlat", axis=1, inplace=True)
baseline.drop("index", axis=1, inplace=True)



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

plt.figure()
plt.plot(xs,ys,label= 'UWB')
plt.plot(xs1,ys1, color = 'red',label='Predicted')
plt.plot(xs2,ys2, color = 'green',label='Baseline')


plt.legend()


plt.figure()
plt.plot(xs,ys,label= 'UWB')
plt.plot(lat,lon, color = 'black',label='GPS')
plt.legend()


plt.show()


"""
plt.figure()

plt.plot(xgps,ygps, color = 'red',label='Raw GPS')
plt.xlabel("longitude")
plt.ylim = (52.01998,52.02016)
plt.xlim = (4.26848,4.26876)
plt.ylabel("latitude")
plt.show()
# https://machinelearningmastery.com/multi-output-regression-models-with-python/

"""
