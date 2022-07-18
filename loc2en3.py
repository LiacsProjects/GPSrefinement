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
    global errorchosen
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

def gettime(ttraject,beacon1,beacon2,beacon3,beacon4):
    Times0 = int(ttraject["second"] + ttraject["minute"]*60 + ttraject["hour"]*60*60)
    Times1 = int(beacon1["second"] + beacon1["minute"]*60 + beacon1["hour"]*60*60)
    Times2 = int(beacon2["second"] + beacon2["minute"]*60 + beacon2["hour"]*60*60)
    Times3 = int(beacon3["second"] + beacon3["minute"]*60 + beacon3["hour"]*60*60)
    Times4 = int(beacon4["second"] + beacon4["minute"]*60 + beacon4["hour"]*60*60)
    tt = [Times0,Times1,Times2,Times3,Times4]
    return tt

def get_seconds(time_str):
    hh, mm, ss = time_str.split(':')
    sss,ns = ss.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(sss)

def vulbeaconcorrecties(beacon1,beacon2,beacon3,beacon4,beacons1,timesbeacon):
    corr = pd.DataFrame(columns=['time','beacon','diflat','diflon','error']) # dataframe for all the corrections per beacon 
    global afstanden,av1,av2,av3,avt ,av4,diflat1,diflat2,diflat3,diflat4,diflon1,diflon2,diflon3,diflon4,x1,x2,x3,x4
    diflat1 =[]
    diflat2 =[]
    diflat3 =[]
    diflat4 =[]
    diflon1 = []
    diflon2 =[]
    diflon3 = []
    diflon4 = []
    x1 =[]
    x2 = []
    x3 =[]
    x4 =[]
    av1 = 0 
    av2 = 0 
    av3 = 0 
    av4 = 0
    avt = 0 
    i = 0 
    for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon1.latitude[index]
        lon = beacon1.longitude[index]
        beaconhere =  (beacons1[0][0],beacons1[0][1])        # beacon 1 locatie 
        punt = (lat,lon)
        x1.append(beacon1.time[index] ) 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[0][0]-lat
        verschillon = beacons1[0][1]-lon
        av1 = av1 + afstand
        diflon1.append(verschillon)
        diflat1.append(verschillat)
        time = beacon1.time[index] + timesbeacon[1]
        new_row = {'time': int(time) , 'beacon':1, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        
    
    for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon2.latitude[index]
        lon = beacon2.longitude[index] 
        beaconhere =  (beacons1[1][0],beacons1[1][1])
        x2.append(beacon2.time[index] )         # beacon 2 locatie 
        punt = (lat,lon)                                     # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[1][0]-lat
        verschillon = beacons1[1][1]-lon
        av2 = av2 + afstand
        diflon2.append(verschillon)
        diflat2.append(verschillat)
        time = beacon2.time[index]+ timesbeacon[2]
        new_row = {'time': int(time) , 'beacon':2, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        
    
    for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon3.latitude[index]
        lon = beacon3.longitude[index] 
        beaconhere =  (beacons1[2][0],beacons1[2][1])        # beacon 0 locatie 
        punt = (lat,lon)  
        x3.append(beacon3.time[index] )                                    # measured point 
        afstand = distance.great_circle(beaconhere,punt).m             
        verschillat = beacons1[2][0]-lat
        verschillon = beacons1[2][1]-lon
        av3 = av3 + afstand
        diflon3.append(verschillon)
        diflat3.append(verschillat)
        time = beacon3.time[index] + timesbeacon[3]
        new_row = {'time': int(time) , 'beacon':3, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
        
   
    for index in beacon4.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
        lat = beacon4.latitude[index]
        lon = beacon4.longitude[index] 
        beaconhere =  (beacons1[3][0],beacons1[3][1])
        x4.append(beacon4.time[index] )         # beacon 0 locatie 
        punt = (lat,lon)                                     # measured point    
        afstand = distance.great_circle(beaconhere,punt).m            
        verschillat = beacons1[3][0]-lat
        verschillon = beacons1[3][1]-lon
        av4 = av4 + afstand
        diflon4.append(verschillon)
        diflat4.append(verschillat)
        time = beacon4.time[index]+ timesbeacon[4]
        new_row = {'time': int(time) , 'beacon':4, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
        corr = corr.append(new_row,ignore_index = True)
    
    av1 = av1 / len(beacon1)
    av2 = av2 / len(beacon2)
    av3 = av3 / len(beacon3)
    av4 = av4 / len(beacon4)
    avt = (av1 + av2 + av3 + av4) / 4
    corr.dropna(subset=["diflat","diflon"],inplace= True)  
    
    return corr

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
traject = pd.read_csv('002 d9d7 2022 06 29 10 56 00 gps.dat',sep =",",skiprows=1) # Read trajectory 
beacon1 = pd.read_csv('020 dae4 2022 06 29 10 46 26 gps.dat',sep =",",skiprows=1) 
beacon2 = pd.read_csv('030 db3a 2022 06 29 10 46 03 gps.dat',sep =",",skiprows=1) 
beacon3 = pd.read_csv('040 daa0 2022 06 29 10 45 20 gps.dat',sep =",",skiprows=1) 
beacon4 = pd.read_csv('050 dc81 2022 06 29 10 45 03 gps.dat',sep =",",skiprows=1)    


columns = ["id","watch","year","month","day","hour","minute","second"]
tra = pd.read_csv('002 d9d7 2022 06 29 10 56 00 gps.dat',sep =" ",nrows=1,names=columns)
bea1 = pd.read_csv('020 dae4 2022 06 29 10 46 26 gps.dat',sep =" ",nrows=1,names=columns)
bea2 = pd.read_csv('030 db3a 2022 06 29 10 46 03 gps.dat',sep =" ",nrows=1,names=columns) 
bea3 = pd.read_csv('040 daa0 2022 06 29 10 45 20 gps.dat',sep =" ",nrows=1,names=columns) 
bea4 = pd.read_csv('050 dc81 2022 06 29 10 45 03 gps.dat',sep =" ",nrows=1,names=columns)    


beacon1 = beacon1[beacon1.accuracy < 64]
beacon2 = beacon2[beacon2.accuracy < 64]
beacon3 = beacon3[beacon3.accuracy < 64]
beacon4 = beacon4[beacon4.accuracy < 64]




beacons2 =[(52.03302,4.25509),(52.03298,4.25481),(52.03313,4.25476),(52.03326,4.25495)]
beacons3=[(52.033563,4.275165),(52.03369,4.27553),(52.033605,4.275717),(52.033435,4.275341)]
#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************

tijd = gettime(tra,bea1,bea2,bea3,bea4)
corrections = vulbeaconcorrecties(beacon1,beacon2,beacon3,beacon4,beacons2,tijd) # change beacons2 to beacons3 when using for location 3
frame = algo1(traject,beacons2,corrections,tijd) # change beacons2 to beacons3 when using for location 3
print(avt)
print(av1,av2,av3,av4)


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

plt.plot(xstraject, ystraject, color='r', linewidth=3, label='Original trajectory')

xscorrectie=[]
yscorrectie=[]
for index in frame.index:                                                             # plotten van het correctie traject 
    puntlat,puntlon = float(frame.latitude[index]),float(frame.longitude[index])
    xpt, ypt =puntlon,puntlat
    plt.plot(xpt, ypt, markersize=1)
    xscorrectie.append(xpt)
    yscorrectie.append(ypt)
    plt.plot(xpt, ypt, 'c*', markersize=2)

plt.plot(xscorrectie, yscorrectie, color='b', linewidth=3, label='Corrected trajectory')

xsgt = [4.27517,4.27553,4.27573,4.27534,4.27517]
ysgt =[52.03356,52.03369,52.03360,52.03343,52.03356]

ysgt2 = [4.25507,4.25500,4.25489,4.25480,4.25471,4.25501]
xsgt2 = [52.03302,52.03304,52.03301,52.03297,52.03302,52.03325]

plt.plot(ysgt2, xsgt2, color='y', linewidth=3, label='Ground truth')

plt.xlabel("longitude")
plt.ylabel("latitude")
img = plt.imread("map1.3.png")
plt.grid()
plt.legend()  

plt.figure() # gekozen beacon laten zien
plt.plot(xb,yb)
plt.xlabel("Time")
plt.ylabel("Chosen beacon")
plt.grid()


plt.figure() # applied error
plt.plot(xb,errorchosen)
plt.xlabel("Time in seconds")
plt.ylabel("Corrected error in meters")
plt.grid()

plt.figure()
plt.subplot(2,1,1)                                                      # plotten van de longitude error
plt.plot(x1, diflon1, color='blue', linewidth = 1, label = 'beacon 1 ')
plt.plot(x2,diflon2,color='orange', linewidth = 1, label = 'beacon 2')
plt.plot(x3,diflon3,color='yellow', linewidth = 1,label = 'beacon 3')
plt.plot(x4,diflon4,color='black', linewidth = 1,label = 'beacon 4')
plt.hlines(0,0,len(x1),color='red',linewidth = 1,label = '0-line')
plt.ylabel("Longitude error")

plt.subplot(2,1,2)                                                         # Plotten van de latitude error 
plt.plot(x1, diflat1, color='blue', linewidth = 1, label = 'beacon 1 ')
plt.plot(x2,diflat2,color='orange', linewidth = 1, label = 'beacon 2')
plt.plot(x3,diflat3,color='yellow', linewidth = 1,label = 'beacon 3')
plt.plot(x4,diflat4,color='black', linewidth = 1,label = 'beacon 4')
plt.hlines(0,0,len(x1),color='red',linewidth = 1,label = '0-line')
plt.ylabel("Latitude error")
plt.xlabel("Time")
plt.legend()
plt.show()
