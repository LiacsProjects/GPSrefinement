
from cProfile import label
from datetime import timedelta
import math
from statistics import mode
from tempfile import tempdir
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

"""
# kalman filter libraries
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import kf_book.book_plots as book_plots
"""


# First all files needed to compute the correction are loaded in, these are max 5 files from beacons and 1 file from the walked path 

 # skiprows invoegen voor het echte werk
traject = pd.read_csv('001 dc81 2022 01 14 11 59 22 gps.dat',sep =",",skiprows=1) # Read trajectory 
beacon1 = pd.read_csv('011 dc81 2022 01 14 11 59 27 gps.dat',sep =",",skiprows=1) 
beacon2 = pd.read_csv('021 dc81 2022 01 14 11 59 54 gps.dat',sep =",",skiprows=1) 
beacon3 = pd.read_csv('031 dc81 2022 01 14 12 00 27 gps.dat',sep =",",skiprows=1) 
beacon4 = pd.read_csv('041 dc81 2022 01 14 12 00 45 gps.dat',sep =",",skiprows=1)    
beacon5 = pd.read_csv('051 dc81 2022 01 14 12 01 17 gps.dat',sep =",",skiprows=1) 

tra = pd.read_csv('001 dc81 2022 01 14 11 59 22 gps.dat',sep =",",nrows=1)
bea1 = pd.read_csv('011 dc81 2022 01 14 11 59 27 gps.dat',sep =",",nrows=1)
bea2 = pd.read_csv('021 dc81 2022 01 14 11 59 54 gps.dat',sep =",",nrows=1) 
bea3 = pd.read_csv('031 dc81 2022 01 14 12 00 27 gps.dat',sep =",",nrows=1) 
bea4 = pd.read_csv('041 dc81 2022 01 14 12 00 45 gps.dat',sep =",",nrows=1)    
bea5 = pd.read_csv('051 dc81 2022 01 14 12 01 17 gps.dat',sep =",",nrows=1) 

#correctionfile = open('correction1.txt',"r+") # bestand met de gecorrigeerde traject punten
#gnnsfile = open('gnss.txt',"r+")

# tuning the dataframes to get more accuracy in graphs 

traject = traject[traject.time > 350.0]
beacon1 = beacon1[beacon1.time > 350.0]
beacon2 = beacon2[beacon2.time > 350.0]
beacon3 = beacon3[beacon3.time > 350.0]
beacon4 = beacon4[beacon4.time > 350.0]
beacon5 = beacon5[beacon5.time > 350.0]

"""
beacon0 = beacon0[beacon0.accuracy < 4]
beacon1 = beacon1[beacon1.accuracy < 4]
beacon2 = beacon2[beacon2.accuracy < 4]
beacon3 = beacon3[beacon3.accuracy < 4]
beacon4 = beacon4[beacon4.accuracy < 4]
beacon5 = beacon4[beacon5.accuracy < 4]

"""

# data.colums =s.strip() for s in data.colums()

# after the files the basic variables are assigned, like some lists of beacon positions for each location 
z =[]
x0 = [] # x0 tot en met x5 zijn de tijdne van de beacons om grafieken mee te plotten 
x1 =[]
x2= []
x3=[]
x4 =[]
x5=[]

count = 0 
small = 1000
afstand = 0
verschil = 0 
afstandna = 0 

diflon = []     # diflon en diflat tot en met diflon5 en diflat 5 worden gebruikt om de verschillen tussen werkelijke locatie en gemeten locatie te onthouden per beacon
diflat = []
diflon1 = []
diflat1 = []
diflon2 = []
diflat2 = []
diflon3 = []
diflat3 = []
diflon4 = []
diflat4 = []
diflon5= []
diflat5 = []  # tot hier gaat dew verschillen tussen de beacons en werkelijke afstand

beacon11 = (52.033576,4.275106) # Testlocation for beacons at location 3 
beacons1=[(52.020093,4.268766),(52.019918,4.268743),(52.0199568,4.268911),(52.019814,4.268948),(52.0198887,4.2692081)] # beacon coordinates per location 1 is location 1 till 4 for location 4 
beacons2=[(),(),(),(),()]
beacons3=[(),(),(),(),()]
beacons4=[(),(),(),(),()]
beacons1side = []   # voor de zijkant van locatie 1 met de uwb test
beaconsleiden = [(52.15538,4.47845)]
lats = [] # gebruiken voor de kalman filter
longs = [] # gebruiken voor de kalman filter
corr = pd.DataFrame(columns=['time','beacon','diflat','diflon','error']) # dataframe for all the corrections per beacon 
correctionframe = pd.DataFrame(columns =['latitude','longitude'])
#correctionfile.write("Latitude,longtitude" + "\n")

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
# section of functions 
# function to compute 

def gettime():
    global ttraject
    global T
    ttraject = tra[5]*60*60 +tra[6]*60+tra[7]
    Tijd1= bea1[5]*60*60 + bea1[6]*60+ bea1[7]
    T1= bea2[5]*60*60 + bea2[6]*60+ bea2[7]
    T2= bea3[5]*60*60 + bea3[6]*60+ bea3[7]
    T3= bea4[5]*60*60 + bea4[6]*60+ bea4[7]
    T4= bea5[5]*60*60 + bea5[6]*60+ bea5[7]
    T = [T0,Tijd1]
    return T


def time(tijdgiven , tijdgiven2,beaconstart):
    seconden1 =  ttraject + tijdgiven
    seconden2 = beaconstart + tijdgiven2
    secondenverschil = abs(seconden1 -seconden2)
    return secondenverschil



# first algorithm 
def  algo1(traject,beaconsf,corrframe):                       # now we can try to find the right correction for each point of gps in the path
    for index in traject.index:                                   # now we can try to find the right correction for each point of gps in the path
        lon = traject.longitude[index] 
        lat = traject.latitude[index]
        tijd = traject.time[index]
        lats.append(lat)
        longs.append(lon)
        point = (lat,lon)
        small = 1000 
        nr = 0
        for x in beaconsf:                                          # for each point in the trajectory find the closest beacon and save that 
            afstand =   distance.great_circle(x,point)
            if afstand < small:
                small = afstand
                beac = x 
                beactime = tijd
                beaconnr = nr 
            nr = nr + 1 
        controltime = 10000
        yb.append(beaconnr)
        xb.append(tijd)
        for row in corr.index:                                    # Lookup the correction for the moment that was closest to the moment of the walking watch
            if corr.beacon[row] == beaconnr:                      # first we need to look up the right beacon rows in the dataframe 
                timedif = abs(corr.time[row]-tijd)                 # then we need to find the right moment so the absolute time difference has to be as less as possible
                if timedif < controltime :                      
                    controltime = timedif
                    correctionlat = corr.diflat[row]
                    correctionlon = corr.diflon[row]
                    errortemp = corr.error[row]
        errorchosen.append(errortemp)
        lat = lat + correctionlat
        lon = lon + correctionlon
        lat = str(lat)
        lon = str(lon)
        new = {'latitude':lat,'longitude': lon}
        corrframe =  corrframe.append(new,ignore_index = True)
    temp = pd.DataFrame(corrframe)
    return temp

# function for getting all gps lines from gnns logger app
def search_string_in_file(file_name, string_to_search):         # https://thispointer.com/python-search-strings-in-a-file-and-get-line-numbers-of-lines-containing-the-string/
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

xanim1 = []
yanim1 = []
xanim2 = []
yanim2 = []
def animate(i):
    xanim1.append(xs[i])
    yanim1.append((ys[i]))
    xanim2.append(xs1[i])
    yanim2.append((ys1[i]))
    plt.plot(xanim1,yanim1, scaley=True, scalex=True, color="red")
    plt.plot(xanim2,yanim2, scaley=True, scalex=True, color="blue")




# from here start to work on beacon and trajectory files 
av =0


for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon1.latitude[index]
    lon = beacon1.longitude[index]
    beaconhere =  (beacons1[0][0],beacons1[0][1])        # beacon 1 locatie 
    punt = (lat,lon)
    x1.append(beacon1.time[index] )
    afstand = distance.great_circle(beaconhere,punt).m             
    z.append(afstand)
    av = av + afstand   
    verschillat = beacons1[0][0]-lat
    verschillon = beacons1[0][1]-lon
    time = beacon1.time[index]
    diflon1.append(verschillon)
    diflat1.append(verschillat)
    new_row = {'time': time , 'beacon':1, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
    corr = corr.append(new_row,ignore_index = True)

av1 = av / len(beacon1)
av = 0 
y1 = np.array(z)
z.clear()


for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon2.latitude[index]
    lon = beacon2.longitude[index] 
    beaconhere =  (beacons1[1][0],beacons1[1][1])        # beacon 2 locatie 
    punt = (lat,lon)                                     # measured point 
    x2.append(beacon2.time[index] )   # x-axis is going to have the timeline 
    afstand = distance.great_circle(beaconhere,punt).m             
    z.append(afstand)
    av = av + afstand 
    verschillat = beacons1[1][0]-lat
    verschillon = beacons1[1][1]-lon
    time = beacon2.time[index]
    diflon2.append(verschillon)
    diflat2.append(verschillat)
    new_row = {'time': time , 'beacon':2, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
    corr = corr.append(new_row,ignore_index = True)

av2 = av / len(beacon2)
av = 0 
y2 = np.array(z)
z.clear()
  
for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon3.latitude[index]
    lon = beacon3.longitude[index] 
    beaconhere =  (beacons1[2][0],beacons1[2][1])        # beacon 0 locatie 
    punt = (lat,lon)                                     # measured point 
    x3.append(beacon3.time[index] )   # x-axis is going to have the timeline 
    afstand = distance.great_circle(beaconhere,punt).m             
    z.append(afstand)
    av = av + afstand 
    verschillat = beacons1[2][0]-lat
    verschillon = beacons1[2][1]-lon
    time = beacon3.time[index]
    diflon3.append(verschillon)
    diflat3.append(verschillat)
    new_row = {'time': time , 'beacon':3, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
    corr = corr.append(new_row,ignore_index = True)

av3 = av / len(beacon3)
av = 0 
y3 = np.array(z)
z.clear()

for index in beacon4.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon4.latitude[index]
    lon = beacon4.longitude[index] 
    beaconhere =  (beacons1[3][0],beacons1[3][1])        # beacon 0 locatie 
    punt = (lat,lon)                                     # measured point 
    x4.append(beacon4.time[index] )   # x-axis is going to have the timeline 
    afstand = distance.great_circle(beaconhere,punt).m             
    z.append(afstand) 
    av = av + afstand
    verschillat = beacons1[3][0]-lat
    verschillon = beacons1[3][1]-lon
    time = beacon4.time[index]
    diflon4.append(verschillon)
    diflat4.append(verschillat)
    new_row = {'time': time , 'beacon':4, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
    corr = corr.append(new_row,ignore_index = True)

av4 = av / len(beacon4)
av = 0 

y4 = np.array(z)
z.clear()

for index in beacon5.index  :      # een eventuele 5 want dan kan er gekeken worden naar als alle beacons op dezelfde locatie staan
    lat = beacon5.latitude[index]
    lon = beacon5.longitude[index] 
    beaconhere =  (beacons1[4][0],beacons1[4][1])        # beacon 0 locatie 
    punt = (lat,lon)                                     # measured point 
    x5.append(beacon5.time[index] )   # x-axis is going to have the timeline 
    afstand = distance.great_circle(beaconhere,punt).m             
    z.append(afstand)      # y -axis is teh dsitance to the real point
    av = av + afstand
    time = beacon5.time[index]
    verschillat = beacons1[4][0] - lat 
    verschillon = beacons1[4][1] - lon 
    diflon5.append(verschillon)
    diflat5.append(verschillat)
    new_row = {'time': time , 'beacon':5, 'diflat': verschillat,'diflon': verschillon,'error': afstand}
    corr = corr.append(new_row,ignore_index = True)

av5 = av / len(beacon5)
av = 0 
avt = (av1 + av2 + av3 + av4 + av5 )/5
y5 = np.array(z)
print(av1,av2,av3,av4,av5)
print(avt)

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************




corr.dropna(subset=["diflat","diflon"],inplace= True)                                   # remove all rows with a nan value in it 
yb = []  
xb = []
errorchosen = []    
#gettime() 
frame = algo1(traject,beacons1,correctionframe)
correctionframe = frame


#algo2(traject,beacons1,correctionframe)






#******************************************************************************************************************************************************************
# maken van alle plots begint hier  
# figure 1                          https://plotly.com/python/hover-text-and-formatting/ mogelijk gebruiken voor hoveren 

#correction_path = 'correction1.txt'                                                    # hier gaat iets fout met inlezen van de file
#data = pd.read_csv(correction_path, names=['latitude', 'longitude'], sep=',')       # het verkrijgen van de data van de correctie
"""
img = plt.imread("Map1.3.png")                                                      # voorbereiden van de achtergrond van de locatie 
fig, ax = plt.subplots()
ax.imshow(img, extent=[4.2683, 4.2694, 52.0197, 52.0202])
"""

xs = [] # latitude van de beacons
ys = [] # longitude van de beacons 
xs1 =[]
ys1 =[]

xsbeacon = []
ysbeacon = []
xsbeacon1 = []
ysbeacon1 = []
xsbeacon2 = []
ysbeacon2 = []
xsbeacon3 = []
ysbeacon3 = []
xsbeacon4 = []
ysbeacon4 = []
xsbeacon5 = []
ysbeacon5 = []

plt.figure()
for index in traject.index:                                                             # plotten van het originele traject 
    puntlat,puntlon = traject.latitude[index],traject.longitude[index]
    xpt, ypt = puntlon,puntlat
    plt.plot(xpt, ypt, markersize=1)
    xs.append(xpt)
    ys.append(ypt)
    plt.plot(xpt, ypt, 'c*', markersize=2)

plt.plot(xs, ys, color='r', linewidth=3, label='Measured trajectory')

for index in correctionframe.index:                                                             # plotten van het correctie traject 
    puntlat,puntlon = float(correctionframe.latitude[index]),float(correctionframe.longitude[index])
    xpt, ypt =puntlon,puntlat
    plt.plot(xpt, ypt, markersize=1)
    xs1.append(xpt)
    ys1.append(ypt)
    plt.plot(xpt, ypt, 'c*', markersize=2)

plt.plot(xs1, ys1, color='b', linewidth=3, label='Corrected trajectory')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
img = plt.imread("map1.3.png")
#plt.imshow(img, zorder= 0,extent=[4.2685, 4.2693, 52.0197,52.0202])
xstraject =[4.26864,4.268766,4.26921,4.26907,4.26895,4.26900,4.26880,4.26875,4.26864]
ystraject = [52.01997,52.020093,52.01986,52.01975,52.01981,52.01985,52.01995,52.01991,52.01997]
plt.plot(xstraject,ystraject,color='y', linewidth=3, label='Estimation true path')
plt.legend()    
plt.grid()

                                                                                # hier eindigt het plotten van de twee trajecten in 1 bestand
                                                                                
plt.figure()              
                                                                  # figure 2 
#plt.subplot(4,1,1)                                                                             # plotting the points 
#plt.plot(x0, y0, color='green', linewidth = 1, label = 'beacon 0 ')
#plt.axhline( y = av0, color='red', linewidth = 1, label = 'beacon 0 av ')


#plt.plot(x0,y0,color='green', linewidth = 1, label = 'beacon 0')
#plt.axhline( y = av1, color='black', linewidth = 1, label = 'beacon 1 av ')

plt.plot(x1,y1,color='blue', linewidth = 1, label = 'beacon 1')
#plt.axhline( y = av1, color='pink', linewidth = 1, label = 'beacon 1 av ')

plt.plot(x2,y2,color='orange', linewidth = 1,label = 'beacon 2')
#plt.axhline( y = av2, color='grey', linewidth = 1, label = 'beacon 2 av ')

plt.plot(x3,y3,color='yellow', linewidth = 1,label = 'beacon 3')
#plt.axhline( y = av3, color='magenta', linewidth = 1, label = 'beacon 3 av ')

plt.plot(x4,y4,color='black', linewidth = 1,label = 'beacon 4')
#plt.axhline( y = av4, color='midnightblue', linewidth = 1, label = 'beacon 4 av ')

plt.plot(x5,y5,color='purple', linewidth = 1,label = 'beacon 5')  
#plt.axhline( y = av5, color='brown', linewidth = 1, label = 'beacon 5 av ')

                                                                                    # naming the x axis
plt.xlabel('Time in seconds')
plt.legend()
                                                                                        # naming the y axis
plt.ylabel('Beaconerror in metres') 
                                                                                        # giving a title to my graph
plt.title('Beaconerror over time ')
 # function to show the plot
plt.savefig('beaconerrorovertime.png')
# hier eindigt het plotten van de beaconcorrecties

plt.figure()                         # plotten van gekozen beaconnr over tijd
plt.plot(xb,yb)
plt.xlabel("Time in seconds")
plt.ylabel("Beaconnr")

plt.figure()
plt.subplot(2,1,1)                                                      # plotten van de longitude error
plt.plot(x1, diflon1, color='blue', linewidth = 1, label = 'beacon 1 ')
plt.plot(x2,diflon2,color='orange', linewidth = 1, label = 'beacon 2')
plt.plot(x3,diflon3,color='yellow', linewidth = 1,label = 'beacon 3')
plt.plot(x4,diflon4,color='black', linewidth = 1,label = 'beacon 4')
plt.plot(x5,diflon5,color='purple', linewidth = 1,label = 'beacon 5')
plt.hlines(0,350,1000,color='red',linewidth = 1,label = '0-line')
plt.ylabel("Longitude error")

plt.subplot(2,1,2)                                                         # Plotten van de latitude error 
plt.plot(x1, diflat1, color='blue', linewidth = 1, label = 'beacon 1 ')
plt.plot(x2,diflat2,color='orange', linewidth = 1, label = 'beacon 2')
plt.plot(x3,diflat3,color='yellow', linewidth = 1,label = 'beacon 3')
plt.plot(x4,diflat4,color='black', linewidth = 1,label = 'beacon 4')
plt.plot(x5,diflat5,color='purple', linewidth = 1,label = 'beacon 5')
plt.hlines(0,350,1000,color='red',linewidth = 1,label = '0-line')
plt.ylabel("Latitude error")
plt.legend()
plt.xlabel("Time")
"""
fig = plt.figure(figsize=(12,8))                                                                        # fig 3 is animatie 
ani = FuncAnimation(fig=fig, func=animate, interval=20)
"""

#******************************************************************************************************************************************************************

#******************************************************************************************************************************************************************

#******************************************************************************************************************************************************************

plt.figure() # showing the error chosen in stead of which beacon was chosen
plt.plot(xb,errorchosen)
plt.xlabel("Time in seconds")
plt.ylabel("Corrected error in metres")


#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
# locatie 4 


#******************************************************************************************************************************************************************

plt.show()

#******************************************************************************************************************************************************************


#locatie1 is 52.0209,4,426783 : 52,01887 , 4.27133,map1
#locatie 2 is ,map2
#locatie 3 is 52.03385,4.27468 : 52.03313,4.27618 , map3
#locatie 4 is ,map4


#correctionfile.close   
#gnnsfile.close

#****************************************************************************************************************************************************************** file end 
"""

vanaf hier wordt het quantitatieve gedeelte gemaakt , om cijfermatig te kunnen bepalen of er werkelijk een verbetering is. 

# vanaf hier  begint het maken van het plaatje 
vis = GPSVis('061 dc81 2012 01 26 02 13 06 gps.dat',map_path='map3.png',points=(52.03385,4.27468, 52.03313,4.27618))  # deze regel invullen wat de waarde zijn van de map en dergelijke 
vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
vis.plot_map(output='save')

print()
#locatie1 is 52.02094,426783 : 52,01887 , 4.27133,map1
#locatie 2 is ,map2
#locatie 3 is 52.03385,4.27468 : 52.03313,4.27618 , map3
#locatie 4 is ,map4
 
"""