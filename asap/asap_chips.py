import json

import ee

def getSignal(request):

	pt = ee.Geometry.Point(float(request.get('lon')), float(request.get('lat')))
    
	cloud_pcent = int(request.get('cloudpcent'))

	deltad = int(request.get('deltad'))
	end_date = ee.Date(request.get('end_period'))
	start_date = end_date.advance(-deltad, 'day')
    
	s2 = ee.ImageCollection('COPERNICUS/S2').filterBounds(pt).\
	  filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent).\
	  filterDate(start_date, end_date).sort('system:time_start')

	s2 = s2.map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))

	chips_date = ee.List(s2.aggregate_array('system:time_start')).map(lambda t: ee.Date(t).format("YYYY-MM-dd"))

	region = pt.transform(projection,1.0).buffer(640.0, ee.ErrorMargin(1.0)).bounds(1.0, projection)

	profile -= s2.getRegion(pt.buffer(20.0, 1.0).bounds(1.0), 10.0)

	print profile.getInfo()

	return None

  
if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '150'}

	profile = getSignal(request)

	print profile

