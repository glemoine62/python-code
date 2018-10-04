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

	projection = ee.Image(s2.first()).select('B2').projection()

	region = pt.transform(projection,1.0).buffer(640.0, ee.ErrorMargin(1.0)).bounds(1.0, projection)


	s2 = s2.map(lambda img: ee.Image(img).reproject(projection).clip(region).visualize(bands=['B8', 'B11', 'B4'], max=[6000, 6000, 4000]))
	#print s2.size().getInfo()
	
	url = ee.data.makeThumbUrl(ee.data.getThumbId({'image':s2.serialize(), 'dimensions': "128x128", 'format': 'png'}))

	#print chips_date.getInfo()
	chips= {'chips_dates': chips_date.getInfo(), 'thurl': url}
	

	return json.dumps(chips)

  
if __name__ == '__main__':

	ee.Initialize()

	request = {'lon': '5.6844', 'lat': '52.7073', 'cloudpcent': '10', 'end_period': '2018-09-30', 'deltad': '150'}

	chip_list = getChips(request)

	print chip_list

"""<!DOCTYPE html><html><head><style>"""

#home {
    width: 128px;
    height: 128px;
    background: url("https://earthengine.googleapis.com/api/thumb?thumbid=5e88ab3c24c90e73a287228d136bb500&token=1d5d77cd397448410db7994433742ff2") 0 0;
}

#next {
    width: 128px;
    height: 128px;
    background: url("https://earthengine.googleapis.com/api/thumb?thumbid=5e88ab3c24c90e73a287228d136bb500&token=1d5d77cd397448410db7994433742ff2") 0 -128px;
}

#last {
	width: 128px;
    height: 128px;
    background: url("https://earthengine.googleapis.com/api/thumb?thumbid=5e88ab3c24c90e73a287228d136bb500&token=1d5d77cd397448410db7994433742ff2") 0 -256px;
}
"""</style></head><body>"""

<img id="home" src="img_trans.png"><nbsp;>
<img id="next" src="img_trans.png"><nbsp;>
<img id="last" src="img_trans.png">
"""</body></html>"""