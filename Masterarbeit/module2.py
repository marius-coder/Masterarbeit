



import numpy as np

import pandas as pd







import pandas as pd
from scipy.stats import *
import scipy
import matplotlib.pyplot as plt

# Load data into a pandas DataFrame
data = pd.read_csv(f"./Gefilterte Daten/Temperatur_OG6.csv", sep=";", decimal=",", encoding= "cp1252")
print(data.info())
# Select the column of interest
#data = data[' T_ATH2GLTH2 [C]'] # T_ATH2GLTH2 [C],  T_ATMWGLTH2 [C],  T_ATP2GLTH2 [C]



data.plot(kind='density', subplots=True, sharex=False)

plt.show()