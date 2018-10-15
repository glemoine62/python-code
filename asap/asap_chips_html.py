import json

import ee

def getChips(request):

	pt = ee.Geometry.Point(float(request.get('lon')), float(request.get('lat')))
    
	cloud_pcent = int(request.get('cloudpcent'))

	deltad = int(request.get('deltad'))
	end_date = ee.Date(request.get('end_period'))
	start_date = end_date.advance(-deltad, 'day')
    
	s2 = ee.ImageCollection('COPERNICUS/S2').filterBounds(pt).\
	  filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent).\
	  filterDate(start_date, end_date).sort('system:time_start')

	chips_date = ee.List(s2.aggregate_array('system:time_start')).map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
	sensor_code = ee.List(s2.aggregate_array('SPACECRAFT_NAME')).map(lambda s: ee.String(s).slice(-2))

	projection = ee.Image(s2.first()).select('B2').projection()

	region = pt.transform(projection,1.0).buffer(320.0, ee.ErrorMargin(1.0)).bounds(1.0, projection)


	s2 = s2.map(lambda img: ee.Image(img).reproject(projection).clip(region).visualize(bands=['B8', 'B11', 'B4'], max=[6000, 6000, 4000]))
	#print s2.size().getInfo()
	
	url = ee.data.makeThumbUrl(ee.data.getThumbId({'image':s2.serialize(), 'dimensions': "128x128", 'format': 'png'}))

	#print chips_date.getInfo()
	chips= {'chips_dates': chips_date.getInfo(), 'sensor_codes': sensor_code.getInfo(), 'thurl': url}
	

	return chips

  
if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '150'}

	chips = getChips(request)

	chip_dates = chips.get('chips_dates')
	sensor_codes = chips.get('sensor_codes')
	#print chip_list

	print """<!DOCTYPE html><html><head><style>"""

	for i in range(len(chip_dates)):
		print "#S{}{} {}\n    width: 128px;\n    height: 128px;\n    background: url(\"{}\") 0 {}px;\n{}".format(sensor_codes[i], chip_dates[i], '{', chips.get('thurl'), -128*i, '}')

	print """</style></head><body>"""

	for i in range(len(chip_dates)):
		print "<div><label>S{}_{}<img id = \"S{}{}\" src=\"img_trans.png\"></label></div><nbsp;>".format(sensor_codes[i], chip_dates[i], sensor_codes[i], chip_dates[i])

	print """</body></html>"""