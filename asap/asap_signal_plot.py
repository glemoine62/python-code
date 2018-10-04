import ee
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt

def getSignal(request):

	pt = ee.Geometry.Point(float(request.get('lon')), float(request.get('lat')))
    
	cloud_pcent = int(request.get('cloudpcent'))

	deltad = int(request.get('deltad'))
	end_date = ee.Date(request.get('end_period'))
	start_date = end_date.advance(-deltad, 'day')
    
	s2 = ee.ImageCollection('COPERNICUS/S2').filterBounds(pt).\
	  filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent).\
	  filterDate(start_date, end_date).sort('system:time_start')

	s2 = s2.map(lambda img: img.addBands([img.normalizedDifference(['B8', 'B4']).rename('NDVI'), ee.Image.constant(img.get('system:time_start')).rename('tstamp')]))

	profile = s2.select(['tstamp', 'NDVI']).map(lambda img: img.set('stats', img.reduceRegion(ee.Reducer.mean(), pt.buffer(20.0, 1.0).bounds(1.0), 10.0)))

	return profile.aggregate_array('stats').getInfo()

  
if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '650'}

	profile = getSignal(request)

	df = DataFrame(profile, columns = ['NDVI', 'tstamp'])

	df['time'] = df.tstamp.map(lambda t: datetime.fromtimestamp(int(t/1e3)))

	df.set_index(['time'], inplace=True)

	print df

	plt.figure()
	plt.plot(df.index, df['NDVI'])
	plt.show()