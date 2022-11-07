#
# Author: Richard van Dijk 
# 
#   Research Software Engineer @ LIACS
#   E: m.k.van.dijk@liacs.leidenuniv.nl 
#

import pandas as pd
import matplotlib.pyplot as plt


#
# Constants
#
Ustarttime = 71                   # UWB starttime traintestset
Uendtime = 367                    # UWB endtime traintestset

#
# Load moving watch GPS W
#
traject = pd.read_csv('051 daa0 2022 03 16 12 41 33 gps.dat',sep =",",skiprows=1)
traject[' latitude'] = traject[' latitude'] - 52.020
traject[' longitude'] = traject[' longitude'] - 4.268

Wt_   = list(traject['time'])
Wlat_ = list(traject[' latitude'])
Wlon_ = list(traject[' longitude'])

# fill the gaps with prior values
Wt__   = []
Wlat__ = []
Wlon__ = []
index = 0
for time in range(round(Wt_[0]), len(Wt_)):
    Wt__.append(time)
    Wlat__.append(Wlat_[index])
    Wlon__.append(Wlon_[index])
    if time == round(Wt_[index]):
        index += 1

# take out time slot
Wt   = []
Wlat = []
Wlon = []
for time in range(Ustarttime, Uendtime+1):
    Wt.append(time)
    Wlat.append(Wlat__[time])
    Wlon.append(Wlon__[time])

plt.figure()
fig,ax = plt.subplots()
plt.plot(Wt,Wlat,label= 'Watch lat')
plt.plot(Wt,Wlon,label= 'Watch lon')
plt.xlabel("time [sec]")
plt.ylabel("GPS decimal")
plt.legend()

plt.figure()
fig,ax = plt.subplots()
plt.plot(Wlat,Wlon,label= 'Watch lat vs lon')
plt.xlabel("latitude decimal")
plt.ylabel("longitude decimal")
plt.legend()


#
# Load the groundtruth measured with ultra wide band technology
#
def get_seconds(time_str):
    hh, mm, ss = time_str.split(':')
    sss,ns = ss.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(sss)

col_list = ["tagId","timestamp","dateTime","loc(x)","loc(y)"]
uwb1 = pd.read_csv("test1.csv",usecols=col_list)
uwb1 = uwb1[uwb1["tagId"] == 26694]

Ut_ = list(uwb1['dateTime'])
Ux_ = list(uwb1['loc(x)'])
Uy_ = list(uwb1['loc(y)'])

# convert time to seconds, offset on start watchtime
Ut__ = []
Ux__ = []
Uy__ = []
for index in range(0, len(Ut_)):
    Ut__.append(get_seconds(Ut_[index]) - 45693)
    Ux__.append(Ux_[index])
    Uy__.append(Uy_[index])
    
# remove the duplicates
Ut_ = []
Ux_ = []
Uy_ = []
time1 = 0
for index in range(0,len(Ut__)):
    time2 = Ut__[index]
    if time1 != time2:
        Ux_.append(Ux__[index])
        Uy_.append(Uy__[index])
        Ut_.append(time2)
        time1 = time2

# fill the gaps with prior values to get a timeslot of Ustarttime:Uendtime
Ut = []
Ux = []
Uy = []
index = 0
for time in range(Ustarttime, Uendtime+1):
    Ut.append(time)
    Ux.append(Ux_[index])
    Uy.append(Uy_[index])
    if time == Ut_[index]:
        index += 1

plt.figure()
fig,ax = plt.subplots()
plt.plot(Ut,Ux,label= 'UWB x')
plt.plot(Ut,Uy,label= 'UWB y')
plt.xlabel("time [sec]")
plt.ylabel("x or y [mm]")
plt.legend()

#
# Take the inner join of U and W, where len(U) < len(W) and Ut = Wt
#
# resulting in Mt, Mx, My, Mlat, Mlon
#
Mt    = []
Mx    = []
My    = []
Mlat  = []
Mlon  = []

# merge W and U -> M
for index in range(0,len(Ut)):
    for index2 in range(0,len(Wt)):
        if Ut[index] == Wt[index2]:
            Mt.append(Ut[index])
            Mx.append(Ux[index]/10000000)
            My.append(Uy[index]/10000000)
            Mlat.append(Wlat[index2])
            Mlon.append(Wlon[index2])

plt.figure()
fig,ax = plt.subplots()
plt.plot(Mt,Mx,label= 'UWB x')
plt.plot(Mt,My,label= 'UWB y')
plt.plot(Mt,Mlat,label= 'Watch lat')
plt.plot(Mt,Mlon,label= 'Watch lon')
plt.xlabel("time [sec]")
plt.ylabel("x or y [meters / 10,000,000] or GPS decimal")
plt.legend()

#
# Load beacon GPS coordinates B[beacon]
#

B1 = pd.read_csv('010 dae4 2022 03 16 12 12 56 gps.dat', sep =",", skiprows=1)
B2 = pd.read_csv('020 db3a 2022 03 16 12 15 34 gps.dat', sep =",", skiprows=1)
B3 = pd.read_csv('030 dae4 2022 03 16 12 13 23 gps.dat', sep =",", skiprows=1)
B4 = pd.read_csv('040 dac4 2022 03 16 12 13 36 gps.dat', sep =",", skiprows=1)

Bt_   = [[],[],[],[]]
Blat_ = [[],[],[],[]]
Blon_ = [[],[],[],[]]

Bt_[0] = list(B1['time'])
Bt_[1] = list(B2['time'])
Bt_[2] = list(B3['time'])
Bt_[3] = list(B4['time'])

Blat_[0] = list(B1['latitude'])
Blat_[1] = list(B2['latitude'])
Blat_[2] = list(B3['latitude'])
Blat_[3] = list(B4['latitude'])

Blon_[0] = list(B1['longitude'])
Blon_[1] = list(B2['longitude'])
Blon_[2] = list(B3['longitude'])
Blon_[3] = list(B4['longitude'])

T = [-1717, -1559, -1690, -1677]
Q = [[52.020161,4.268634],[52.020093,4.268767],[52.019981,4.268616],[52.020048,4.268484]]

# remove the first digits of the GPS coordinates and round the time
Bt__    = [[],[],[],[]]
Blat__  = [[],[],[],[]]
Blon__  = [[],[],[],[]]
for index1 in range(0,4):
    for index2 in range(0,len(Bt_[index1])):
        Bt__[index1].append(round(Bt_[index1][index2]) + T[index1])
        Blat__[index1].append(Blat_[index1][index2] - 52.020)
        Blon__[index1].append(Blon_[index1][index2] - 4.268)
    Q[index1][0] -= 52.020
    Q[index1][1] -=  4.268

# remove the duplicates
Bt_     = [[],[],[],[]]
Blat_   = [[],[],[],[]]
Blon_   = [[],[],[],[]]
for index1 in range(0,4):
    time1 = Bt__[index1][0]
    for index2 in range(0,len(Bt__[index1])):
        time2 = Bt__[index1][index2]
        if time1 != time2:
            Blat_[index1].append(Blat__[index1][index2])
            Blon_[index1].append(Blon__[index1][index2])
            Bt_[index1].append(time2)
            time1 = time2

# fill the gaps with prior values
Bt   = [[],[],[],[]]
Blat = [[],[],[],[]]
Blon = [[],[],[],[]]
for index1 in range(0,4):
    for index2 in range(0,len(Bt_[index1])):
        if Bt_[index1][index2] == Ustarttime:
            break
    for time in range(Ustarttime, Uendtime+1):
        Bt[index1].append(time)
        Blat[index1].append(Blat_[index1][index2])
        Blon[index1].append(Blon_[index1][index2])
        if time == Bt_[index1][index2]:
            index2 += 1

# calculate difference lat, lon and distance
from math import sqrt

Bdlat = [[],[],[],[]]
Bdlon = [[],[],[],[]]
Bdst  = [[],[],[],[]]
for index1 in range(0,4):
    for index2 in range(0, len(Bt[index1])):
        dlat = Q[index1][1] - Blat[index1][index2]
        dlon = Q[index1][0] - Blon[index1][index2]
        Bdlat[index1].append(dlat)
        Bdlon[index1].append(dlon)
        Bdst[index1].append(sqrt(dlat*dlat + dlon*dlon))

plt.figure()
fig,ax = plt.subplots()
for index in range(0,4):
    slabel = "Blat" + str(index+1)
    plt.plot(Bt[index], Blat[index], label=slabel)
for index in range(0,4):
    slabel = "Blon" + str(index+1)
    plt.plot(Bt[index],Blon[index],label=slabel)
plt.xlabel("time [sec]")
plt.ylabel("GPS [decimal]")
plt.legend()

plt.figure()
fig,ax = plt.subplots()
for index in range(0,4):
    slabel = "Bdst" + str(index+1)
    plt.plot(Bt[index], Bdst[index], label=slabel)
plt.xlabel("time [sec]")
plt.ylabel("sqrt(dlat+dlon) [gps decimal]")
plt.legend()

plt.figure()
fig,ax = plt.subplots()
for index in range(0,4):
    slabel = "Blat vs lon" + str(index+1)
    plt.plot(Blat[index], Blon[index], label=slabel)
plt.xlabel("latitude [gps decimal]")
plt.ylabel("longitude [gps decimal]")
plt.legend()

plt.figure()
fig,ax = plt.subplots()
for index in range(0,4):
    slabel = "Bdif lat vs lon" + str(index+1)
    plt.plot(Bdlat[index], Bdlon[index], label=slabel)
plt.xlabel("diff latitude [gps decimal]")
plt.ylabel("diff longitude [gps decimal]")
plt.legend()


#
# Merge the four beacons with time with Mt and Bt
#
# resulting in Mdlat[0:3], Mdlon[0:3], besides already calculated Mt, Mx, My, Mlat, Mlon
#
Mdlat  = [[],[],[],[]]
Mdlon  = [[],[],[],[]]

# merge Bdlat and Bdlon
for index1 in range(0,4):
    for index2 in range(0,len(Bt[index1])):
        if Mt[index2] == Bt[index1][index2]:
            Mdlat[index1].append(Bdlat[index1][index2])
            Mdlon[index1].append(Bdlon[index1][index2])

#
# Save result to file
#

traintestdata = {
  "time"   : Mt,
  "wlat"   : Mlat,
  "wlon"   : Mlon,
  "b1dlat" : Mdlat[0],
  "b1dlon" : Mdlon[0],
  "b2dlat" : Mdlat[1],
  "b2dlon" : Mdlon[1],
  "b3dlat" : Mdlat[2],
  "b3dlon" : Mdlon[2],
  "b4dlat" : Mdlat[3],
  "b4dlon" : Mdlon[3],
  "uwbx"   : Mx,
  "uwby"   : My
}

df = pd.DataFrame(traintestdata)
df.to_csv("traintestset.csv")
