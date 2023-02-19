# -*- coding: cp1252 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

data = pd.read_csv("Filtered.csv", sep=";", decimal=",", encoding= "cp1252",parse_dates=["Datetime"],index_col='Datetime')
data.index = pd.to_datetime(data.index, format = '%d.%m.%Y %H:%M:%S')
print(data.info())


toPlot = ["Kühldecke_DS2","Büro_BA1","Kühldecke_6OG_Bereich_1","Kühldecke_6OG_Bereich_2","Kühldecke_DS1",
          "Kühlregister_L01","Kühlregister_L02","Kühlregister_L03","Konferenz","Halle_Lobby","Küche","LAN_Räume"]
toPlot = ["Summe_Verbrauch","Summe_Erzeugung","Differenz","Außentemperatur"]
data[toPlot].info()
fig, ax = plt.subplots()
data[toPlot].resample("d").mean().plot(kind="line", ax=ax);

ax.set_title('World population')
ax.set_xlabel('Year')
ax.set_ylabel('Number of people (millions)')

plt.show()

