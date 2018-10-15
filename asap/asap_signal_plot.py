import ee
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import asap_signal as asig

  
if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '350'}
	request = {'lon': '34.36357', 'lat': '7.86203', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '350'}
 
	moprofile = asig.getMODISSignal(request)

	df = DataFrame(moprofile, columns = ['NDVI', 'tstamp'])

	df['time'] = df.tstamp.map(lambda t: datetime.fromtimestamp(int(t/1e3)))

	df.set_index(['time'], inplace=True)

	s1profile = asig.getS1Signal(request)

	df1 = DataFrame(s1profile, columns = ['VH', 'VV', 'tstamp'])

	df1['time'] = df1.tstamp.map(lambda t: datetime.fromtimestamp(int(t/1e3)))

	df1.set_index(['time'], inplace=True)

	fig, ax1 = plt.subplots()
	ax1.plot(df.index, df['NDVI'], marker = '^', linestyle = ' ', color = 'blue', label = "MODIS NDVI")
	ax1.set_xlabel('time (s)')

	ax1.set_ylabel('NDVI', color='blue')	
	ax1.set_ylim(0, 1)
	
	ax2 = ax1.twinx()
	ax2.set_ylabel('$\sigma^\degree$ (dB)', color='red')	

	ax2.plot(df1.index, 10.0*np.log10(df1['VV']), marker = 'o', linestyle = ' ', color = 'red', label = "S1 VV")
	ax2.plot(df1.index, 10.0*np.log10(df1['VH']), marker = 'x', linestyle = ' ', color = 'green', label = "S1 VH")
	ax2.plot(df1.index, 10.0*np.log10(df1['VH']/df1['VV']), marker = 'x', linestyle = ' ', color = 'black', label = "S1 VH/VV")
	ax2.set_ylim(-25, 0)

	fig.legend()
	fig.tight_layout()

	plt.show()
	

