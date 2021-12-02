from datetime import timedelta
import math
import pandas as pd
import numpy as np
import matplotlib as plt
from pandas.io.formats.format import Timedelta64Formatter
from PIL import Image, ImageDraw
from gps_class import GPSVis


data = pd.read_csv('testgps.dat', sep =",") # skiprows invoegen voor het echte werk
# data.colums =s.strip() for s in data.colums()

a = 6.378 # m 
b = 6.356 # m
G = b**2

# het bereken van ECEF coordinaten 
#data = data[data.private != 'P']
for index in data.index :      # loop gebruiken voor ecef coordinaten 
    tlat = data.latitude[index]
    tlon = data.longitude[index]
    lat = math.radians(tlat)
    lon = math.radians(tlon)
    H = 0
    I = math.sin(lat)**2
    J = math.cos(lat)**2

    N = a**2 / math.sqrt(a**2 * J + b**2 * I)

    T = N + H 

    X = T * math.cos(lat)*math.cos(lon)
    Y = T * math.cos(lat) * math.sin(lon)

    Z = ((b*b / a * a)/ T) * math.sin(lat)
    #print(X,Y,Z)

# vanaf hier  begint het maken van het plaatje 
vis = GPSVis('testgps.dat',map_path='map2.png',points=(52.3163,4.5346, 52.2583,4.6086))  # deze regel invullen wat de waarde zijn van de map en dergelijke 
vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
vis.plot_map(output='save')

print()


#controle op coordinaten

 #function measure(lat1, lon1, lat2, lon2){  // generally used geo measurement function
  #  var R = 6378.137; // Radius of earth in KM
   # var dLat = lat2 * Math.PI / 180 - lat1 * Math.PI / 180;
    #var dLon = lon2 * Math.PI / 180 - lon1 * Math.PI / 180;
    #var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    #Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    #Math.sin(dLon/2) * Math.sin(dLon/2);
    #var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    #var d = R * c;
    #return d * 1000; // meters
#}