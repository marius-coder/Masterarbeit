# -*- coding: cp1252 -*-
import pandas as pd
import numpy as np
import datetime as dt
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def DetermineDay(time) -> str:
	"""Findet den Typ des Tages heraus. Achtet dabei auf Feiertage.
	hour: int
		Stunde des Jahres"""
	import holidays
	Feiertage = holidays.country_holidays('Austria')
	ret = []	
	for step in time:
		if step in Feiertage:
			#Feiertage werden als Sonntage gehandhabt
			ret.append(0)
		if time[hour].weekday() == 5:
			ret.append(0)
		elif time[hour].weekday() == 6:
			ret.append(0)
		else:
			ret.append(1)


