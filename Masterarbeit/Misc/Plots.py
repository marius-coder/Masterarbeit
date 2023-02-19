# -*- coding: cp1252 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

data = pd.read_csv("Filtered.csv", sep=";", decimal=",", encoding= "cp1252",parse_dates=["Datetime"],index_col='Datetime')
data.index = pd.to_datetime(data.index, format = '%d.%m.%Y %H:%M:%S')
print(data.info())


toPlot = ["K�hldecke_DS2","B�ro_BA1","K�hldecke_6OG_Bereich_1","K�hldecke_6OG_Bereich_2","K�hldecke_DS1",
          "K�hlregister_L01","K�hlregister_L02","K�hlregister_L03","Konferenz","Halle_Lobby","K�che","LAN_R�ume"]
toPlot = ["Summe_Verbrauch","Summe_Erzeugung","Differenz","Au�entemperatur"]
data[toPlot].info()
fig, ax = plt.subplots()
data[toPlot].resample("d").mean().plot(kind="line", ax=ax);

ax.set_title('World population')
ax.set_xlabel('Year')
ax.set_ylabel('Number of people (millions)')

plt.show()

