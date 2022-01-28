from datetime import timedelta
import math
import pandas as pd
import numpy as np
import matplotlib as plt
from pandas.io.formats.format import Timedelta64Formatter
from PIL import Image, ImageDraw
from gps_class import GPSVis
from geographiclib.geodesic import Geodesic
from geopy import distance 

# First all files needed to compute the correction are loaded in, these are max 5 files from beacons and 1 file from the walked path 

 # skiprows invoegen voor het echte werk
traject = pd.read_csv('010 d9d7 2012 02 06 07 01 44 gps.dat',sep =",") # Read trajectory 
beacon0 = pd.read_csv('020 dc81 2012 01 29 01 47 30 gps.dat',sep =",") # read beacon files 
beacon1 = pd.read_csv('030 daa0 2012 01 29 01 48 16 gps.dat',sep =",") 
beacon2 = pd.read_csv('040 db3a 2012 01 29 09 26 30 gps.dat',sep =",") 
beacon3 = pd.read_csv('050 dae4 2012 01 29 10 14 33 gps.dat',sep =",") 
beacon4 = pd.read_csv('060 dac4 2012 01 28 22 55 03 gps.dat',sep =",") 
correctionfile = open('correction.txt',"r+") # bestand met de gecorrigeerde traject punten
# data.colums =s.strip() for s in data.colums()

# after the files the basic variables are assigned, like some lists of beacon positions for each location 
count = 0 
small = 1000
afstand = 0
verschil = 0 
afstandna = 0 
beacon11 = (52.033576,4.275106) # Testlocation for beacons at location 3 
beacons1=[(52.0200495,4.2687917),(52.0199295,4.268824),(52.0199568,4.268911),(52.0198563,4.2689535),(52.0198726,4.2961596)] # beacon coordinates per location 1 is location 1 till 4 for location 4 
beacons2=[(),(),(),(),()]
beacons3=[(),(),(),(),()]
beacons4=[(),(),(),(),()]
corr = pd.DataFrame(columns=['time','beacon','diflat','diflon']) # dataframe for all the corrections per beacon 
correctionfile.write("Latitude,longtitude" + "\n")

for index in beacon0.index  :      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon0.latitude[index]
    lon = beacon0.longitude[index] 
    time = beacon0.time[index]
    verschillat = beacons1[0][0] - lat 
    verschillon = beacons1[0][1] - lon 
    new_row = {'time': time , 'beacon':0, 'diflat': verschillat,'diflon': verschillon}
    corr = corr.append(new_row,ignore_index = True)

for index in beacon1.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon1.latitude[index]
    lon = beacon1.longitude[index] 
    verschillat = beacons1[1][0]-lat
    verschillon = beacons1[1][1]-lon
    time = beacon1.time[index]
    new_row = {'time': time , 'beacon':1, 'diflat': verschillat,'diflon': verschillon}
    corr = corr.append(new_row,ignore_index = True)

for index in beacon2.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon2.latitude[index]
    lon = beacon2.longitude[index] 
    verschillat = beacons1[2][0]-lat
    verschillon = beacons1[2][1]-lon
    time = beacon2.time[index]
    new_row = {'time': time , 'beacon':2, 'diflat': verschillat,'diflon': verschillon}
    corr = corr.append(new_row,ignore_index = True)
  
for index in beacon3.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon3.latitude[index]
    lon = beacon3.longitude[index] 
    verschillat = beacons1[3][0]-lat
    verschillon = beacons1[3][1]-lon
    time = beacon3.time[index]
    new_row = {'time': time , 'beacon':3, 'diflat': verschillat,'diflon': verschillon}
    corr = corr.append(new_row,ignore_index = True)

for index in beacon4.index:      # alle bestanden van de beacons afgaan om de correcties te zetten in dataframe
    lat = beacon4.latitude[index]
    lon = beacon4.longitude[index] 
    verschillat = beacons1[4][0]-lat
    verschillon = beacons1[4][1]-lon
    time = beacon4.time[index]
    new_row = {'time': time , 'beacon':4, 'diflat': verschillat,'diflon': verschillon}
    corr = corr.append(new_row,ignore_index = True)
                                                                                        # all beaconcorrections are now saved in corr as dataframe 
corr.dropna(subset=["diflat","diflon"],inplace= True)                                   # remove all rows with a nan value in it 
print(corr)
                                                                                        # now we can try to find the right correction for each point of gps in the path
for index in traject.index:      # Alle punten van het traject afgaan en bekijken welke beacon het dichtsbij is
    lon = traject.longitude[index] 
    lat = traject.latitude[index]
    tijd = traject.time[index]
    point = (lat,lon)
    small = 1000 
    nr = 0
    for x in beacons1:                                          # for each point in the trajectory find the closest beacon and save that 
        afstand =   distance.great_circle(x,point)
        if afstand < small:
            small = afstand
            beac = x 
            beactime = tijd
            beaconnr = nr 
        nr = nr + 1 
    controltime = 20
    for row in corr.index:                                    # Lookup the correction for the moment that was closest to the moment of the walking watch
        if corr.beacon[row] == beaconnr:                      # first we need to look up the right beacon rows in the dataframe 
            timedif = abs(corr.time[row]-tijd)                 # then we need to find the right moment so the absolute time difference has to be as less as possible
            if timedif < controltime :                      
                controltime = timedif
                correctionlat = corr.diflat[row]
                correctionlon = corr.diflon[row]
    lat = lat + correctionlat
    lon = lon + correctionlon
    lat = str(lat)
    lon = str(lon)
    correctionfile.write(lat + "," + lon +"\n")



       

  
correctionfile.close   

""""
traject = traject[traject["accuracy"]<="4.0"] # filter away outliers in the trajectory 

for index in traject.index:    # alle punten uit het traject doorlopen 
    lat = traject.latitude[index]
    lon = traject.longitude[index]
    tijd = traject.time[index]
    nieuw3 = (lat,lon)
    small = 10000
    for beacon in beacons1:                                 # determine all distances to each beacon
        distance = distance.great_circle(beacon,nieuw3).m  # nieuw3 is het punt uit gemeten uit het gelopen traject
        if distance < small:                                # if the distance is smaller then the smallest found before use this one 
            small = distance
            use = beacon                                     # now we found and saved the right beacon 
                                                            # next we have to take the right correction of the error of the beacon, herefore we 




for index in data.index:     
    lat = data.latitude[index]
    lon = data.longitude[index] 
    verschillat = beacon3[0]-lat
    verschillon = beacon3[1]-lon
    latn = lat + verschillat 
    lonn = lon + verschillon 
    nieuw2 = (latn, lonn)
    afstandna = afstandna +  distance.great_circle(beacon3,nieuw2 ).m
    small = 10000
    for beacon in beacons1:                  # determine all distances to each beacon
        distance = distance.great_circle(beacon,nieuw3).m  # nieuw3 is het punt uit gemeten uit het gelopen traject
        if distance < small:                 # if the distance is smaller then the smallest found before use this one 
            small = distance
            use = beacon 
            diflon = beacons1[beacon[1]] - data.longitude[index]
            diflat = beacons1[beacon[0]] - data.latitude[index]

                                             # if we know which beacon was near then we save the values needed to compute the correction 
                                             # after this we use our correction method described below to correct the points measured
                                             # if distances are equal pick the beacon with the highest accuracy at that moment

# Calculateaccuracy difference for beacon files 
#data = data[data.private != 'P']

for index in data.index :      # Use loop to compute average error of a single beacon 
    tlat = data.latitude[index]
    tlon = data.longitude[index]
    punt = (tlat,tlon)
    afstand = afstand + distance.great_circle(beacon3,punt ).m    # Use this fucntion to compute distance over globe 
    count = count + 1
gem = afstand / count
print(gem)
print(afstand)
print(count)


"""

"""
opzet voor het bepalen van welke beacon gebruikt moet worden
for index in data.index:                     # Per Point determine which beacon is the nearest one 
    small = 10000
    for beacon in beacons1:                  # determine all distances to each beacon
        distance = distance.great_circle(beacon,nieuw3).m
        if distance < small:                 # if the distance is smaller then the smallest found before use this one 
            small = distance
            use = beacon 
            diflon = beacons1[beacon[1]] - data.longitude[index]
            diflat = beacons1[beacon[0]] - data.latitude[index]
                                             # if we know which beacon was near then we save the values needed to compute the correction 
                                             # after this we use our correction method described below to correct the points measured
                                             # if distances are equal pick the beacon with the highest accuracy at that moment

"""
"""
from here the try out correction method is staged.   
elif verschillon > 0 and verschillat > 0:    
        latn = lat - verschillat 
        lonn = lon - verschillon 
        nieuw3 = (latn, lonn)
        afstandna2 = afstandna2 +  distance.great_circle(beacon3,nieuw3).m
    elif verschillat < 0 and verschillon >0  :
        latn = lat + verschillat
        lonn = lon - verschillon
        nieuw4 = (latn, lonn)
        afstandna3 = afstandna3 +  distance.great_circle(beacon3,nieuw4).m
    elif verschillat > 0 and verschillon < 0 :
        latn = lat - verschillat
        lonn = lon + verschillon
        nieuw5 = (latn, lonn)
        afstandna4 = afstandna4 +  distance.great_circle(beacon3,nieuw5).m
    else:                               # to check if there are any other possibilites for now 
        print("niet")

print(afstandna2)
print(afstandna3)
print(afstandna4)

tot hier ging het corrigeren van de data voor de beacon    

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
