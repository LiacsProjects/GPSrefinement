from datetime import timedelta
import math
from statistics import mode
from tempfile import tempdir
from turtle import color
from matplotlib.font_manager import json_load
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import animation 
from matplotlib.animation import FuncAnimation
from itertools import count
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn
from pandas.io.formats.format import Timedelta64Formatter
from PIL import Image, ImageDraw
#from gps_class import GPSVis
from geographiclib.geodesic import Geodesic
from geopy import distance 
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection


def  algo1(traject,beaconsf,corrframe,alltimes): 
    global yb 
    yb =[]
    global xb
    xb =[]  
    lats =[]
    longs =[]
    errorchosen = []
    corr= []
    global beaconchosen
    for index in traject.index:                                   # now we can try to find the right correction for each point of gps in the path
        lon = traject.longitude[index] 
        lat = traject.latitude[index]
        tijd = traject.time[index] + alltimes[0]
        lats.append(lat)
        longs.append(lon)
        point = (lat,lon)
        small = np.Infinity
        nr = 1
        for x in beaconsf:                                          # for each point in the trajectory find the closest beacon and save that 
            afstand =   distance.great_circle(x,point)
            if afstand < small:
                small = afstand
                beac = x 
                beactime = tijd
                beaconnr = nr 
            nr = nr + 1 
        controltime = np.Infinity
        yb.append(beaconnr)
        xb.append(tijd)
        for row in corrframe.index:                                    # Lookup the correction for the moment that was closest to the moment of the walking watch
            if corrframe.beacon[row] == beaconnr:
                timedif = abs(corrframe.time[row]-tijd)                 # then we need to find the right moment so the absolute time difference has to be as less as possible
                if timedif < controltime :
                    controltime = timedif
                    correctionlat = corrframe.diflat[row]
                    correctionlon = corrframe.diflon[row]
                    errortemp = corrframe.error[row]
        errorchosen.append(errortemp)
        lat = lat + correctionlat
        lon = lon + correctionlon
        lat = str(lat)
        lon = str(lon)
        new = {'latitude':lat,'longitude': lon}
        corr.append(new)
    temp = pd.DataFrame(corr)
    return temp

def gettime(ttraject,beacon1,beacon2,beacon3):
    Times0 = int(ttraject["second"] + ttraject["minute"]*60 + ttraject["hour"]*60*60)
    Times1 = int(beacon1["second"] + beacon1["minute"]*60 + beacon1["hour"]*60*60)
    Times2 = int(beacon2["second"] + beacon2["minute"]*60 + beacon2["hour"]*60*60)
    Times3 = int(beacon3["second"] + beacon3["minute"]*60 + beacon3["hour"]*60*60)
    tt = [Times0,Times1,Times2,Times3]
    return tt

def get_seconds(time_str):
    hh, mm, ss = time_str.split(':')
    sss,ns = ss.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(sss)

def vulbeaconcorrecties(beacon1,beacon2,beacon3,beacons1,timesbeacon):
    corr = pd.DataFrame(columns=['time','beacon','diflat','diflon','error']) # dataframe for all the corrections per beacon 
    global afstanden,av1,av2,av3,avt
    av1 = 0 
    av2 = 0 
    av3 = 0 
    avt = 0 
    i = 0 
    for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon1.latitude[index]
        lon = beacon1.longitude[index]
        beaconhere =  (beacons1[0][0],beacons1[0][1])        # beacon 1 locatie 
        punt = (lat,lon)
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[0][0]-lat
        verschillon = beacons1[0][1]-lon
        time = beacon1.time[index] + timesbeacon[1]
        new_row = {'time': int(time) , 'beacon':1, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        av1 = av1 + afstand
    
    for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon2.latitude[index]
        lon = beacon2.longitude[index] 
        beaconhere =  (beacons1[1][0],beacons1[1][1])        # beacon 2 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[1][0]-lat
        verschillon = beacons1[1][1]-lon
        time = beacon2.time[index]+ timesbeacon[2]
        new_row = {'time': int(time) , 'beacon':2, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        av2 = av2 + afstand
    
    for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon3.latitude[index]
        lon = beacon3.longitude[index] 
        beaconhere =  (beacons1[2][0],beacons1[2][1])        # beacon 0 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[2][0]-lat
        verschillon = beacons1[2][1]-lon
        time = beacon3.time[index] + timesbeacon[3]
        new_row = {'time': int(time) , 'beacon':3, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        av3 = av3 + afstand

    av1 = av1 / len(beacon1)
    av2 = av2 / len(beacon2)
    av3 = av3 / len(beacon3)
    avt = (av1+av2+av3) / 3
    corr.dropna(subset=["diflat","diflon"],inplace= True)  
        
    return corr

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
traject = pd.read_csv('002 d9d7 2022 05 09 10 12 11 gps.dat',sep =",",skiprows=1) # Read trajectory 
beacon1 = pd.read_csv('012 dae4 2022 05 09 10 11 45 gps.dat',sep =",",skiprows=1) 
beacon2 = pd.read_csv('022 db3aaccelerometer_interval_ms_int 2022 05 09 10 15 40 gps.dat',sep =",",skiprows=1) 
beacon3 = pd.read_csv('032 dc81 2022 05 09 10 11 06 gps.dat',sep =",",skiprows=1) 



columns = ["id","watch","year","month","day","hour","minute","second"]
tra = pd.read_csv('002 d9d7 2022 05 09 10 12 11 gps.dat',sep =" ",nrows=1,names=columns)
bea1 = pd.read_csv('012 dae4 2022 05 09 10 11 45 gps.dat',sep =" ",nrows=1,names=columns)
bea2 = pd.read_csv('022 db3aaccelerometer_interval_ms_int 2022 05 09 10 15 40 gps.dat',sep =" ",nrows=1,names=columns) 
bea3 = pd.read_csv('032 dc81 2022 05 09 10 11 06 gps.dat',sep =" ",nrows=1,names=columns) 



beacons4 =[(52.03552,4.27340),(52.03566,4.27360),(52.03535 ,4.27369)] # starting left top with 1 then clockwise 2 and 3

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
tijd = gettime(tra,bea1,bea2,bea3)
corrections = vulbeaconcorrecties(beacon1,beacon2,beacon3,beacons4,tijd)
frame = algo1(traject,beacons4,corrections,tijd)
print(avt)
print(av1,av2,av3)


#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************

xstraject =[]
ystraject =[]
plt.figure()

for index in traject.index:                                                             # plotten van het originele traject 
    puntlat,puntlon = traject.latitude[index],traject.longitude[index]
    xpt, ypt = puntlon,puntlat
    plt.plot(xpt, ypt, markersize=1)
    xstraject.append(xpt)
    ystraject.append(ypt)
    plt.plot(xpt, ypt, 'c*', markersize=2)

plt.plot(xstraject, ystraject, color='r', linewidth=3, label='Measured trajectory')

xscorrectie=[]
yscorrectie=[]
for index in frame.index:                                                             # plotten van het correctie traject 
    puntlat,puntlon = float(frame.latitude[index]),float(frame.longitude[index])
    xpt, ypt =puntlon,puntlat
    plt.plot(xpt, ypt, markersize=1)
    xscorrectie.append(xpt)
    yscorrectie.append(ypt)
    plt.plot(xpt, ypt, 'c*', markersize=2)

plt.plot(xscorrectie, yscorrectie, color='b', linewidth=3, label='Corrected Trajectory')
xgroundtruth = [52.03552,52.03566,52.03546,52.03534,52.03552]
ygroundtruth = [4.27341,4.27362,4.27368,4.27369,4.27341]

plt.plot(ygroundtruth,xgroundtruth,color= 'yellow',linewidth=3, label='Ground truth')

plt.xlabel("longitude")
plt.ylabel("latitude")
img = plt.imread("map1.3.png")
plt.grid()
plt.legend()  

plt.figure() # gekozen beacon laten zien
plt.plot(xb,yb)
plt.xlabel("Time")
plt.ylabel("Chosen beacon")

plt.show()
