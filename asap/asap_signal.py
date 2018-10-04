import ee

def getS2Signal(request):

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

def getL8Signal(request):

	pt = ee.Geometry.Point(float(request.get('lon')), float(request.get('lat')))
    
	cloud_pcent = int(request.get('cloudpcent'))

	deltad = int(request.get('deltad'))
	end_date = ee.Date(request.get('end_period'))
	start_date = end_date.advance(-deltad, 'day')
    
	l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT').filterBounds(pt).\
	  filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_pcent).\
	  filterDate(start_date, end_date).sort('system:time_start')

	l8 = l8.map(lambda img: img.addBands([img.normalizedDifference(['B5', 'B4']).rename('NDVI'), ee.Image.constant(img.get('system:time_start')).rename('tstamp')]))

	profile = l8.select(['tstamp', 'NDVI']).map(lambda img: img.set('stats', img.reduceRegion(ee.Reducer.mean(), pt.buffer(50.0, 1.0).bounds(1.0), 30.0)))

	return profile.aggregate_array('stats').getInfo()


def getMODISSignal(request):

	pt = ee.Geometry.Point(float(request.get('lon')), float(request.get('lat')))
    
	deltad = int(request.get('deltad'))
	end_date = ee.Date(request.get('end_period'))
	start_date = end_date.advance(-deltad, 'day')

	mo = ee.ImageCollection('MODIS/006/MOD13Q1').filterBounds(pt).\
	  filterDate(start_date, end_date).sort('system:time_start')

	mo = mo.map(lambda img: img.divide(10000.0).addBands([ee.Image.constant(img.get('system:time_start')).rename('tstamp')]))

	profile = mo.select(['tstamp', 'NDVI']).map(lambda img: img.set('stats', img.reduceRegion(ee.Reducer.mean(), pt.buffer(100.0, 1.0).bounds(1.0), 250.0)))

	return profile.aggregate_array('stats').getInfo()  

if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '350'}

	dict_profile = {'S2': getS2Signal(request)} 
	dict_profile['L8'] = getL8Signal(request)
	dict_profile['MODIS'] = getMODISSignal(request)

	print dict_profile

