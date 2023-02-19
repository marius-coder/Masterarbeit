# -*- coding: cp1252 -*-



import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import holidays
Feiertage = holidays.country_holidays('Austria')
if False:
    files = glob.glob("./Daten/*.csv")
    #for datei in glob.glob("./Daten/*.csv"):
    #	data = pd.read_csv(datei, sep= ";", decimal= ",", encoding= "cp1252",on_bad_lines="error")

	

    combined_csv = pd.concat([pd.read_csv(f, sep = ";",on_bad_lines="error",encoding="cp1252",
                                        decimal = ",") for f in files ])



    combined_csv.to_csv( "combined.csv", index=False, encoding='cp1252', decimal = ",", sep = ";")

def IsWorkday(step):
    if step in Feiertage:
        #Feiertage werden als Sonntage gehandhabt
        return 0
    elif step.weekday() == 5:
        return 0
    elif step.weekday() == 6:
        return 0
    else:
        return 1

def DetermineDay(time) -> str:
    """Findet den Typ des Tages heraus. Achtet dabei auf Feiertage.
    hour: int
	    Stunde des Jahres"""
    ret = []	
    for step in time:
        ret.append(IsWorkday(step))
    return ret
    
def DetermineOccupancy(time):
    ret = []	
    occupancy = [0.544264067, 0.448930465, 0.353292422, 0.26702818, 0.184923945, 0.072775177, 0, 0.058738632, 0.165187087, 0.418158045, 0.756184133, 1, 0.980244368, 0.817146945, 0.781567983, 0.705681545, 0.624904251, 0.75220578, 0.863803449, 0.991562126, 0.963958643, 0.851565256, 0.740463163, 0.624633282]

    for step in time:
        if IsWorkday(step) == 1:
            ret.append(occupancy[step.hour])
        else:
            ret.append(0)

    return ret

names = {
    "Radiatoren_6OG_Bereich_1" : ["P058"],
    "Radiatoren_6OG_Bereich_2" : ["P059"],
    "Heizregister_L01" : ["P068"],
    "Nachheizregister_L01" : ["P069"],
    "Kältemaschine_1" : ["P073"],
    "Kältemaschine_2" : ["P074"],
    "Kältemaschine_3" : ["P075"],
    "Free_Cooling" : ["P076"],
    "Kühldecke_6OG_Bereich_1" : ["P078"],
    "Kühldecke_6OG_Bereich_2" : ["P079"],
    "Kühldecke_DS1" : ["K003"],
    "Kühldecke_DS2" : ["P080"],
    "LAN_Räume" : ["P083"],
    "Büro_BA1" : ["L002"],
    "Kühlregister_L01" : ["L001_KR"],
    "Kühlregister_L02" : ["L002_KR"],
    "Kühlregister_L03" : ["L003_KR"],
    "Konferenz" : ["P087"],
    "Halle_Lobby" : ["P088"],
    "Küche" : ["P089"],
    #"CO2_6OG" : ["CO2_OG6"],
    "Temperatur_OG6" : ["T_OG6"],
    "SollTemperatur_Heizen_OG6" : ["HZ_OG6"], #Sind Ventilstellungen Werte von 0-100
    "SollTemperatur_Kühlen_OG6" : ["KUEHL_OG6"], #Sind Ventilstellungen Werte von 0-100
    "CO2_15OG" : ["CO2_OG15","CO2_OG6"],
    #"CO2_Aussenluft" : ["CO2_OG22"],
    "Licht" : ["[lx]"],
    "Aussendaten" : ["LTH2","aH_L01_B02 [g/kg]","CO2_OG22"],
    }

combined_csv = pd.read_csv("combined.csv", sep=";", decimal=",", encoding= "cp1252")

dfTraining = pd.DataFrame()
dfTraining["Kühldecke_DS1"] = combined_csv[" P_K003_KD_P07 [kW]"].clip(upper=10000)
dfTraining["Kühldecke_DS1"][0:3200] = 0
dfTraining["Kühldecke_DS2"] = combined_csv[" P_K005_KD_P080 [kW]"].clip(upper=10000)
dfTraining["Kühldecke_DS2"][0:2600] = 0
dfTraining["LAN_Räume"] = combined_csv[" P_K004_KW_P083 [kW]"].clip(upper=10000)
dfTraining["LAN_Räume"][0:3200] = 0
dfTraining["Büro_BA1"] = combined_csv[" P_L002_KR_P084 [kW]"].clip(upper=10000)
dfTraining["Büro_BA1"][0:3300] = 0
dfTraining["Kühlregister_L01"] = combined_csv[" P_L001_KR_P086 [kW]"].clip(upper=10000)
dfTraining["Kühlregister_L01"][0:3300] = 0
dfTraining["Kühlregister_L02"] = combined_csv[" P_L002_KR_P084 [kW]"].clip(upper=10000)
dfTraining["Kühlregister_L02"][0:2600] = 0
dfTraining["Kühlregister_L03"] = combined_csv[" P_L003_KR_P085 [kW]"].clip(upper=10000)
dfTraining["Kühlregister_L03"][0:2600] = 0
dfTraining["Konferenz"] = combined_csv[" P_L004_KR_P087 [kW]"].clip(upper=10000)
dfTraining["Konferenz"][0:2600] = 0
dfTraining["Halle_Lobby"] = combined_csv[" P_L005_KR_P088 [kW]"].clip(upper=10000)
dfTraining["Halle_Lobby"][0:2600] = 0
dfTraining["Küche"] = combined_csv[" P_L051_KR_P089 [kW]"].clip(upper=10000)
dfTraining["Küche"][0:2570] = 0



dfTraining["Außentemperatur"] = combined_csv[" T_ATMWGLTH2 [C]"].clip(upper=50)
dfTraining["Außentemperatur"][0:2570] = 0
dfTraining["Außenfeuchte"] = combined_csv[" aH_L01_B02 [g/kg]"].clip(upper=10000)
dfTraining["Außenfeuchte"][0:2600] = 0
combined_csv[" CO2_OG22_ABL01 [ppm]"][0:2600] = 0
combined_csv[" CO2_OG22_ABL02 [ppm]"][0:2600] = 0
#inter = combined_csv[" CO2_OG22_ABL01 [ppm]"] - combined_csv[" CO2_OG22_AUL02 [ppm]"]
dfTraining["CO2_OG15_RAW"] = combined_csv[" CO2_OG15_R01_RRZ02 [ppm]"].fillna(combined_csv[" CO2_OG6_R01_RRZ01 [ppm]"]).astype(float)
dfTraining["CO2_1_RAW"] = combined_csv[" CO2_OG22_ABL01 [ppm]"]
dfTraining["CO2_2_RAW"] = combined_csv[" CO2_OG22_ABL02 [ppm]"]
dfTraining["CO2_AUL_RAW"] = combined_csv[" CO2_OG22_AUL02 [ppm]"]

dfTraining["CO2_OG15"] = dfTraining["CO2_OG15_RAW"] / dfTraining["CO2_OG15_RAW"].max()
dfTraining["CO2_AUL"] = combined_csv[" CO2_OG22_AUL02 [ppm]"] / combined_csv[" CO2_OG22_AUL02 [ppm]"].max()
dfTraining["CO2_1"] = combined_csv[" CO2_OG22_ABL01 [ppm]"] / combined_csv[" CO2_OG22_ABL01 [ppm]"].max()
dfTraining["CO2_2"] = combined_csv[" CO2_OG22_ABL02 [ppm]"] / combined_csv[" CO2_OG22_ABL02 [ppm]"].max()

#dfTraining["CO2_3"] = 
dfTraining = dfTraining.clip(lower=0)


# define regular expression pattern
pattern = r'^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$'

# use str.match() to check which rows match the pattern
combined_csv[combined_csv.columns[-2]] = combined_csv[combined_csv.columns[-2]].astype(str)
mask = combined_csv[combined_csv.columns[-2]].str.match(pattern)

# print rows that do not match the pattern
combined_csv[combined_csv.columns[-2]] = combined_csv[combined_csv.columns[-2]].mask(~mask, np.nan)


combined_csv[combined_csv.columns[0]] = combined_csv[combined_csv.columns[0]]+":00"
dfTraining["Datetime"] = pd.to_datetime(combined_csv[combined_csv.columns[0]].fillna(combined_csv[combined_csv.columns[-2]]), format = '%d.%m.%Y %H:%M:%S')
dfTraining = dfTraining.dropna(subset=['Datetime'])
dfTraining = dfTraining.drop_duplicates(subset='Datetime')
dfTraining = dfTraining.set_index(dfTraining["Datetime"])
dfTraining = dfTraining.resample("1min").interpolate("ffill")
dfTraining['Stunde'] = dfTraining.index.hour
print(len(dfTraining))
dfTraining = dfTraining.dropna(subset=["Kühldecke_DS2","Büro_BA1","Kühldecke_DS1","Kühlregister_L01","Kühlregister_L02","Kühlregister_L03","Konferenz","Halle_Lobby","Küche","LAN_Räume"],how="any")
dfTraining["Summe_Verbrauch"] = dfTraining["Kühldecke_DS2"]+dfTraining["Büro_BA1"]+\
                    dfTraining["Kühldecke_DS1"]+dfTraining["Kühlregister_L01"]+dfTraining["Kühlregister_L02"]+dfTraining["Kühlregister_L03"]+\
                    dfTraining["Konferenz"]+dfTraining["Halle_Lobby"]+dfTraining["Küche"]+dfTraining["LAN_Räume"]
dfTraining["Summe_Erzeugung"] =  combined_csv[" P_KM01_KW_P073 [kW]"] + combined_csv[" P_KM02_KW_P074 [kW]"] + combined_csv[" P_KM03_KW_P075 [kW]"]
dfTraining["Differenz"] =  dfTraining["Summe_Erzeugung"] - dfTraining["Summe_Verbrauch"]

dfTraining["Werktag"] = DetermineDay(dfTraining["Datetime"])
dfTraining["Anwesenheit"] = DetermineOccupancy(dfTraining["Datetime"])
dfTraining = dfTraining[3300:]
print(len(dfTraining))
#dfTraining["Datetime"] = pd.date_range(start = pd.to_datetime('02.08.2021 23:59',  dayfirst = True), periods = len(dfTraining), freq = 'min')
dfTraining.to_csv(f"./Filtered.csv", sep=";", decimal=",", encoding="cp1252") # aH_L01_B02 [g/kg] außenfeuchte
dfTraining.resample("15min").mean().to_csv(f"./Filtered_15min.csv", sep=";", decimal=",", encoding="cp1252")
dfTraining.resample("1h").mean().to_csv(f"./Filtered_1hour.csv", sep=";", decimal=",", encoding="cp1252")
