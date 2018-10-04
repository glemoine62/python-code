########################################################################################################################
##      JRC ASAP high resolution module containing:                                                                    #
##          - methods for time series and image chip extraction from GEE                                               #
##          - compositing functions                                                                                    #
##      (c) Patrick Griffiths, August 2018, patrickdgriffiths@gmail.com                                                #
##                                                                                                                     #
##                                                                                                                     #
##                                                                                                                     #
########################################################################################################################

import ee
import os
import csv
import ogr, osr
import gdal
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
#import datetime
import json
import ogr
#import GEE_TMS as gtms
import urllib
try:
    from scipy.misc import bytescale
except:
    from scipy.misc.pilutil import bytescale
from scipy.misc import toimage
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects

class PointOfInterest:

    def __init__(self, name=None, UTM_X=None, UTM_Y=None, LATLON_X=None, LATLON_Y=None,startDate=None, endDate=None):
        self.name = name
        self.UTM_X = UTM_X
        self.UTM_Y = UTM_Y
        self.LATLON_X = LATLON_X
        self.LATLON_Y = LATLON_Y
        self.startDate = startDate
        self.endDate = endDate


def get_UTM_zone_from_longitude(longitude):
    assert longitude <= 180, ""
    return int(math.floor((longitude + 180.)/6.)+1)

def get_EPSG_for_UTM(latitude, zone):
    # to determine UTM epsgs code use 326%% for North zones and 327%% for south zones
    return int('326{0}'.format(zone)) if latitude >= 0.0 else int('327{0}'.format(zone))

def convert_utm_to_latlon(utm_x, utm_y, src_epsg=None, trg_epsg=4326, trim=False):

    assert type(trg_epsg) is int

    source = osr.SpatialReference()
    source.ImportFromEPSG(src_epsg)

    target = osr.SpatialReference()
    target.ImportFromEPSG(trg_epsg)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.CreateGeometryFromWkt("POINT ({0} {1})".format(utm_x, utm_y))
    point.Transform(transform)

    x = float(point.ExportToWkt().split(' ')[1][1:])
    y = float(point.ExportToWkt().split(' ')[2][:-1])

    if trim:
        x = x - int(str(x)[len(str(x))-1])
        y = y - int(str(y)[len(str(y))-1])

    return [x,y]

def convert_latlon_to_utm(lat, lon, src_epsg=4326, trg_epsg=None, trim=True):

    assert type(trg_epsg) is int

    source = osr.SpatialReference()
    source.ImportFromEPSG(src_epsg)

    target = osr.SpatialReference()
    target.ImportFromEPSG(trg_epsg)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.CreateGeometryFromWkt("POINT ({0} {1})".format(lon, lat))
    point.Transform(transform)

    x = int(round(float(point.ExportToWkt().split(' ')[1][1:])))
    y = int(round(float(point.ExportToWkt().split(' ')[2][:-1])))

    if trim:
        x = x - int(str(x)[len(str(x))-1])
        y = y - int(str(y)[len(str(y))-1])

    return [x,y]

def convert_shp2geojson(fp_shp, outfile=None):
    # https://gist.github.com/AlexArcPy/2fc9f41ca164f76fcbb30ebca273b59f
    assert os.path.isfile(fp_shp), 'file does not exist'
    assert fp_shp.endswith('.shp'), 'fp_shp needs to be a shapefile'

    if outfile is None:
        outfile = fp_shp[:-4]+'.json'
    else:
        assert outfile.endswith('.json'), 'outfile needs to be of type *.json'

    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(fp_shp, 0)

    fc = {
        'type': 'FeatureCollection',
        'features': []
        }

    fc = {
        "type": "Feature",
        "properties": {
            "title": "Polygon",
            "id": "polygon"
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": []
        }}

    lyr = data_source.GetLayer(0)
    for feature in lyr:
        #fc['features'].append(feature.ExportToJson(as_object=True))
        fc['geometry']['coordinates'].append(feature.ExportToJson(as_object=True))

    with open(outfile, 'wb') as f:
        json.dump(fc, f)

##### https://groups.google.com/d/topic/google-earth-engine-developers/8vFOJ_Vo8RY/discussion
#### For a shapefile, return a dictionary of ee.Feature
def shapefile_2_ftr_dict_func(input_path, reverse=True):

    input_ds = ogr.Open(input_path)
    input_lyr = input_ds.GetLayer()
    input_osr = input_lyr.GetSpatialRef()

    ## Earth Engine Spatial Reference
    ee_osr = osr.SpatialReference()
    ee_osr.ImportFromEPSG(4326)

    ## Transform to EE Spatial Reference
    ee_tx = osr.CoordinateTransformation(input_osr, ee_osr)

    ## Read feature/geometry and save in dictionary with FID as key
    ee_ftr_dict = dict()
    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        input_fid = input_ftr.GetFID()
        input_geom = input_ftr.GetGeometryRef()

        ## Project geometry
        input_geom.Transform(ee_tx)

        ## Convert feature to GeoJSON
        input_json = json.loads(input_ftr.ExportToJson())

        ## Reverse the point order of the JSON coordinates
        if reverse:
            input_json['geometry']['coordinates'] = [
                list(reversed(item))
                for item in input_json['geometry']['coordinates']]

        ## pg edit, ee expects feature ID to be str...
        if not isinstance(input_json['id'], str):
            input_json['id'] = str(input_json['id'])

        ## Convert feature to GeoJSON then to EE feature
        ee_ftr_dict[input_fid] = ee.Feature(input_json)
        input_geom = None
        input_ftr = input_lyr.GetNextFeature()

    input_ds = None
    return ee_ftr_dict[0]

def offset_utm_coord(x=None, y=None, dist=None, ps=20):
    new_x = (x + dist)
    new_y = (y - dist) #+ ps
    return [new_x, new_y]

def offset_latlon_coords(lat, lon, dist):

    '''
    returns [lat, lon]
    :param lat:
    :param lon:
    :param dist:
    :return:
    '''

    if dist == 0:
        return [lat, lon]

    assert type(lat) == float
    assert type(lon) == float

    const_lat = 110540
    const_lon = 111320

    theta1 = 180
    dx1 = dist * math.sin(math.radians(theta1))
    dy1 = dist * math.cos(math.radians(theta1))

    deltaLon1 = dx1 / (const_lon * math.cos(math.radians(lat)))
    deltaLat1 = dy1 / const_lat

    P1_lon = lon + deltaLon1
    P1_lat = lat + deltaLat1

    theta2 = 90
    dx2 = dist * math.sin(math.radians(theta2))
    dy2 = dist * math.cos(math.radians(theta2))

    deltaLon2 = dx2 / (const_lon * math.cos(math.radians(lat)))
    deltaLat2 = dy2 / const_lat

    P2_lon = lon + deltaLon2
    P2_lat = lat + deltaLat2

    return [P1_lat, P2_lon]

def timestamp_to_datetime(ts):
    if isinstance(ts, list):
        dt = [np.datetime64(int(d), 'ms') for d in ts]
    else:
        dt = np.datetime64(int(ts), 'ms')
    return dt

def datetime2doy(time_arr_sorted):
    years = time_arr_sorted.astype('datetime64[Y]')#.astype(int)
    months = time_arr_sorted.astype('datetime64[M]').astype(int) % 12 + 1
    days = ((time_arr_sorted.astype('datetime64[D]') - time_arr_sorted.astype('datetime64[M]')) + 1).astype(int)
    doys = list()
    for i in range(len(days)):
        idoy = date(int(str(years[i])), months[i], days[i]).strftime("%j")
        doys.append(int(idoy))
    return doys

def get_monthly_ranges(from_date, to_date):

    import calendar

    months = np.arange(from_date, to_date, dtype='datetime64[M]')
    range1 = ["{0}-{1}".format(str(m),"01") for m in list(months)]
    range2 = ["{0}-{1}".format(str(m), calendar.monthrange(int(str(m)[:4]), int(str(m)[5:]))[1]) for m in list(months)]

    return zip(range1, range2)



def clean_EE_obs_data(obs):
    if obs is None:
        return obs
    dc = list()
    for iob in obs[1:]:
        clean = [v if v != None else 0.00 for v in iob]
        dc.append(clean)
    dc.insert(0,obs[0])
    return dc

def sort_imageChips(chipList):
    image_chips_sorted = []

    try:
        chips_datetime_arr = np.array([d.date_time for d in chipList])
    except:
        # if a nested list is returned
        chips_datetime_arr = np.array([d[0].date_time for d in chipList])

    inds = chips_datetime_arr.argsort()

    for i, iChip in enumerate(chipList):
        image_chips_sorted.append(chipList[inds[i]])

    return image_chips_sorted

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    # https://stackoverflow.com/questions/49322932/scipy-implementation-of-savitzky-golay-filter

    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    firstvals = 2 * y[0] - y[1:half_window + 1][::-1] #https://stackoverflow.com/questions/49322932/scipy-implementation-of-savitzky-golay-filter
    lastvals = 2 * y[-1] - y[-half_window - 1:-1][::-1]
    # values taken from the signal itself
    # firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    # lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def reduceImageChipTimeSeries(imageChips, monthly_chips=3, verbose=False):

    assert monthly_chips == 1 or monthly_chips == 3, "current monthly chip selection for one or three chips only"

    date_times = np.array([c.date_time for c in imageChips])
    years = np.array([d.astype(object).year for d in date_times])
    months = np.array([d.astype(object).month for d in date_times])
    days = np.array([d.astype(object).day for d in date_times])

    # select 2-3 per month
    selection=[]

    if monthly_chips == 1:
        for year in np.unique(years):
            for month in range(1,13):
                iMonth = np.where((months == month)&(years == year))
                i15th = None
                if len(iMonth[0]) == 0:
                    continue # skip month
                elif len(iMonth[0]) >= 1:
                    ddiffs = days[iMonth]
                    i15th = min(np.where(abs(15 - ddiffs) == min(abs(15 - ddiffs)))[0])
                    itmp = np.where(ddiffs == ddiffs[i15th])  # in case there are several observations of same day
                    ddiffs[itmp] = 99
                    selection.append(imageChips[iMonth[0][i15th]])
                    if verbose is True:
                        print "days in month {0}-{1}: {2}".format(month, year, days[iMonth[0]])
                        print "day selected: {0}".format(days[i15th])
                        print('')

    elif monthly_chips == 3:
        for year in np.unique(years):
            for month in range(1,13):
                iMonth = np.where((months == month)&(years == year))
                i10th = None
                i20th = None
                i30th = None
                i1st = None
                i2nd = None
                if len(iMonth[0]) == 0:
                    # skip month
                    continue
                elif len(iMonth[0]) >= 3:
                    # select chip that days are closest to day 10, 20, 30; select first if multiple

                    ddiffs = days[iMonth]

                    i10th = min(np.where(abs(10 - ddiffs) == min(abs(10 - ddiffs)))[0])
                    itmp = np.where(ddiffs == ddiffs[i10th]) # in case there are several observations of same day
                    ddiffs[itmp] = 99


                    i20th = min(np.where(abs(20 - ddiffs) == min(abs(20 - ddiffs)))[0])
                    itmp = np.where(ddiffs == ddiffs[i20th])  # in case there are several observations of same day
                    ddiffs[itmp] = 99

                    i30th = min(np.where(abs(30 - ddiffs) == min(abs(30 - ddiffs)))[0])

                    # assert selected dates are in chrono order
                    order = np.array([i10th, i20th, i30th])[np.array([i10th, i20th, i30th]).argsort()]
                    i10th = order[0]
                    i20th = order[1]
                    i30th = order[2]

                    selection.append(imageChips[iMonth[0][i10th]])
                    selection.append(imageChips[iMonth[0][i20th]])
                    selection.append(imageChips[iMonth[0][i30th]])

                    if verbose is True:
                        print "days in month {0}-{1}: {2}".format(month, year, days[iMonth])
                        print "days selected: {0}, {1}, {2}".format(days[iMonth][i10th],days[iMonth][i20th],days[iMonth][i30th])

                elif len(iMonth[0]) == 2:
                    i1st = iMonth[0][0]
                    i2nd = iMonth[0][1]
                    selection.append(imageChips[i1st])
                    selection.append(imageChips[i2nd])
                    if verbose is True:
                        print "days in month {0}-{1}: {2}".format(month, year, days[iMonth[0]])
                        print "days selected: {0}, {1}".format(days[i1st], days[i2nd])
                else:
                    selection.append(imageChips[iMonth[0][0]])
                    if verbose is True:
                        print "days in month {0}-{1}: {2}".format(month, year, days[iMonth[0][0]])

    return selection

def check_sensor_obs_period(date, sensor):
    '''
    Utility function to check if a given sensor was actively aquiring data at a given date
    :param date:
    :param sensor:
    :return:
    '''

    assert sensor in ['S2','L8','L7','L5'], 'unknown sensor: %s' % sensor

    try:
        date_datetime = np.datetime64(date)
    except:
        raise Exception, 'could not convert date to datetime64: %s' % date

    if sensor == 'S2':
        check = True if date_datetime >= np.datetime64('2015-06-23') else False
    elif sensor == 'L8':
        check = True if date_datetime >= np.datetime64('2013-04-11') else False
    elif sensor == 'L7':
        check = True if date_datetime >= np.datetime64('1999-01-01') else False
    elif sensor == 'L5':
        check = True if date_datetime >= np.datetime64('1984-01-01') and date_datetime <= np.datetime64('2012-05-05') else False

    # check = True if sensor == 'S2' and date_datetime >= np.datetime64('2015-06-23') else False
    # check = True if sensor == 'L8' and date_datetime >= np.datetime64('2013-04-11') else False
    # check = True if sensor == 'L7' and date_datetime >= np.datetime64('1999-01-01') else False
    # check = True if sensor == 'L5' and date_datetime >= np.datetime64('1984-01-01') and date_datetime <= np.datetime64('2012-05-05') else False

    return check

def map_RGB_to_bands(rgb_bands, sensor, get_names=False):
    '''
    provide list of three desired bands to display as RGB
    :param rgb_bands: string identifying spectral band such as 'nir', 'swir1'
                      if rgb_bands == 'all' all bands will be returned for a given sensor (currently excluding cirus, coastal and pan)
    :return:
    '''



    assert sensor in ['S2','L8','L7','L5']
    if rgb_bands == 'all': rgb_bands = ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir1','swir2']
    assert isinstance(rgb_bands, list) and len(rgb_bands) >= 3, 'at least three bands expected'
    for b in rgb_bands: assert b in ['blue','green','red','rededge1','rededge2','rededge3','nir','swir1','swir2', 'QA','wv'], 'unknown band: {}'.format(b)

    ignore_bands = ['QA']

    # S2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8a', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60']

    lut = {}
    lut['blue'] =           {'S2' : 'B2', 'L8' : 'B2', 'L7' : 'B1', 'L5' : 'B1'}
    lut['green'] =          {'S2' : 'B3', 'L8' : 'B3', 'L7' : 'B2', 'L5' : 'B2'}
    lut['red']  =           {'S2' : 'B4', 'L8':  'B4', 'L7' : 'B3', 'L5' : 'B3'}
    lut['rededge1'] =       {'S2' : 'B5', 'L8' : None, 'L7' : None, 'L5' : None}
    lut['rededge2'] =       {'S2' : 'B6', 'L8' : None, 'L7' : None, 'L5' : None}
    lut['rededge3'] =       {'S2' : 'B7', 'L8' : None, 'L7' : None, 'L5' : None}
    lut['nir'] =            {'S2' : 'B8', 'L8' : 'B5', 'L7' : 'B4', 'L5' : 'B4'}
    lut['swir1'] =          {'S2' : 'B11','L8':  'B6', 'L7' : 'B5', 'L5' : 'B5'}
    lut['swir2'] =          {'S2' : 'B12','L8':  'B7', 'L7' : 'B7', 'L5' : 'B7'}
    lut['wv'] =             {'S2' : 'B9', 'L8':  None, 'L7': None}
    # lut['nir'] =          {'S2' : 'B2', 'L8':  'B2', 'L7': 'B4'}

    band_identifiers = list()
    if get_names: names = []

    for b in rgb_bands:

        if b in ignore_bands:
            continue

        band_id = lut[b][sensor]

        if band_id is not None:
            band_identifiers.append(band_id)
            if get_names: names.append(b)

    if get_names:
        return band_identifiers, names
    else:
        return band_identifiers

def get_individual_imageChips(x=None, y=None, target_epsg=None, start_date=None, end_date=None,
                              rgb_combination=['nir', 'swir1', 'red'], windowSize=200, pixel_size=20, monthly_chips=1,
                              use_HOT_Mask=True, sensors=['S2', 'L8', 'L7', 'L5'],
                              maxCloud=100, hot_crit=0.02, HOT_cutoff=50, POI_ID=None, outdir=r'c:'):

        start_time_total = time.time()
        # determine if location is lat/lon or UTM
        if (int(x) and int(y)) <= 180:
            use_latlon = True
            use_UTM = False
            lat = y
            lon = x
            utm_x = None
            utm_y = None
        else:
            use_latlon = False
            use_UTM = True
            assert target_epsg is not None, 'for UTM a EPSG is required'
            lat = None
            lon = None
            utm_x = x
            utm_y = y

        if use_latlon:
            zone = get_UTM_zone_from_longitude(lon)
            target_epsg = get_EPSG_for_UTM(latitude=lat, zone=zone)
            utm_xy = convert_latlon_to_utm(lat=lat, lon=lon, trg_epsg=target_epsg)
            lat = None
            lon = None
            utm_x = utm_xy[0]
            utm_y = utm_xy[1]
            use_latlon = False
            use_UTM = True

        available_sensors = ['S2', 'L8', 'L7', 'L5']
        collections = ['COPERNICUS/S2', 'LANDSAT/LC08/C01/T1_TOA', 'LANDSAT/LE07/C01/T1_TOA', 'LANDSAT/LT05/C01/T1_TOA']

        if use_HOT_Mask is True:
            if 'red' not in rgb_combination:
                rgb_combination.append('red')
            if 'blue' not in rgb_combination:
                rgb_combination.append('blue')

        image_chips = []
        rgb_initial = None

        for iCol, sensor in enumerate(available_sensors):

            if sensor in sensors:

                # in case QA has been added previously to rgb_comb, remove it
                if 'QA' in rgb_combination:
                    ipop = rgb_combination.index('QA')
                    rgb_combination.pop(ipop)

                if rgb_initial is not None: rgb_combination = list(rgb_initial)

                # ensure band 9 for new s2 cloudmasking
                if sensor == 'S2':
                    if rgb_initial is None: rgb_initial = list(rgb_combination) # make a safe copy
                    if 'wv' not in rgb_combination: rgb_combination.append('wv')
                    if 'swir1' not in rgb_combination: rgb_combination.append('swir1')
                    if 'nir' not in rgb_combination: rgb_combination.append('nir')
                    if 'red' not in rgb_combination: rgb_combination.append('red')
                    if 'blue' not in rgb_combination: rgb_combination.append('blue')
                else:
                    if 'wv' in rgb_combination:
                        # for other sensors make sure to remove wv
                        ipop = rgb_combination.index('wv')
                        rgb_combination.pop(ipop)

                gee_band_ids = map_RGB_to_bands(rgb_combination, sensor)

                ChipData = eeRetrieval(collection=collections[iCol],
                                       maxCloud=maxCloud,
                                       startDate=start_date,
                                       endDate=end_date,
                                       lat=lat, lon=lon,
                                       utm_x=utm_x, utm_y=utm_y,
                                       epsg=target_epsg,
                                       rectangle=True, rectDist=windowSize,
                                       pixel_size=pixel_size,
                                       bandList=gee_band_ids,
                                       bandNames=rgb_combination,
                                       )

                ChipData.define_geometry()
                ChipData.get_data()

                try:
                    if len(ChipData.retrieval) > 1:
                        chips = ExtractImageChipsFromEERetrieval(ChipData, nb=len(rgb_combination), render_chip=True,outdir=outdir,
                                                                 rgb_combination=rgb_combination,
                                                                 use_HOT_Mask=True, HOT_crit=hot_crit,
                                                                 HOT_cutoff=HOT_cutoff, verbose=False)
                        image_chips.extend(chips)
                except:
                    if ChipData.retrieval is None:
                        continue

                    elif ChipData.retrieval is False:
                        monthly_ranges = get_monthly_ranges(from_date=start_date, to_date=end_date)
                        for mrange in monthly_ranges:

                            if check_sensor_obs_period(mrange[1], sensor) is False:
                                continue

                            if True:
                                print 'retrieving {}-{} for {}'.format(mrange[0], mrange[1], sensor)

                            m_start_data = mrange[0]
                            m_end_data = mrange[1]
                            ChipData = eeRetrieval(collection=collections[iCol],
                                                   maxCloud=maxCloud,
                                                   startDate=m_start_data,
                                                   endDate=m_end_data,
                                                   lat=lat, lon=lon,
                                                   utm_x=utm_x, utm_y=utm_y,
                                                   epsg=target_epsg,
                                                   rectangle=True, rectDist=windowSize,
                                                   pixel_size=pixel_size,
                                                   bandList=gee_band_ids,
                                                   bandNames=rgb_combination,
                                                   )

                            ChipData.define_geometry()
                            ChipData.get_data()

                            if ((ChipData.retrieval is None) or (ChipData.retrieval is False)):
                                continue

                            chips = ExtractImageChipsFromEERetrieval(ChipData, nb=len(rgb_combination), render_chip=True, outdir=outdir,
                                                                     rgb_combination=rgb_combination,
                                                                     use_HOT_Mask=True, HOT_crit=hot_crit,
                                                                     HOT_cutoff=HOT_cutoff, verbose=False)
                            image_chips.extend(chips)

        # optional - sort and rename chips
        # bns = np.array([os.path.basename(fp) for fp in image_chips],dtype=str)
        # sort_index = bns.argsort()
        # image_chips_sorted = image_chips.sort()
        # for i, fp in enumerate(image_chips_sorted):
        #     os.rename(fp, os.path.join(os.path.dirname(fp),'{0}_{1}'.format(i+1, os.path.basename(fp))))

        print 'done'
        return image_chips

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

def create_TimeSeries_Plot(x=None, y=None, target_epsg=None, start_date=None, end_date=None,
                           rgb_combination = ['nir','swir1','red'], windowSize = 200, pixel_size = 20, monthly_chips=1,
                           profile_only=False, profile_mean=False, profile_window=False, use_HOT_Mask=True,
                           sensors = ['S2', 'L8', 'L7','L5'],overplot_modis=True, filtered_profile=False, csv_output=False, csv_only=False,
                           maxCloud=100, hot_crit=0.02, HOT_cutoff=50, POI_ID=None, outdir=r'c:'):
    '''

    :param x: x location, as decimal degree lat or UTM X
    :param y: y location, as decimal degree lon or UTM Y
    :param target_epsg: if UTM then EPSG has to be defined
    :param start_date: start date of time series
    :param end_date: end date of time series
    :param rgb_combination: RGB band combination for image chips, band identified as styring e.g. 'red' or 'nir', this is then translated to sensor band (e.g. 'B04')
                            Known band identifier: 'blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir1','swir2'
    :param windowSize: window size for image chip in meters
    :param pixel_size: pixel size for image chip in meters
    :param monthly_chips: if many chips exist for time series, reduce to this number of chips per month
                          set to None to use all chips
    :param profile_only: do not produce image chips and plots only the time series profile is csv_only is False
    :param profile_mean: True/False: average time series profile
    :param profile_window: window size of average time series profile, has to be uneven number e.g. 7 or 3
    :param use_HOT_Mask: use the Haze Optimized Transformation as a proxy for cloud coverage
    :param sensors:
    :param overplot_modis:
    :param filtered_profile: Sav Golay filter applied to NDVI time series
    :param csv_output:
    :param maxCloud:
    :param hot_crit:
    :param HOT_cutoff:
    :param POI_ID:
    :param outdir:
    :return:
    '''

    start_time_total = time.time()
    # determine if location is lat/lon or UTM
    if (int(x) and int(y)) <= 180:
        use_latlon = True
        use_UTM = False
        lat = y
        lon = x
        utm_x = None
        utm_y = None
    else:
        use_latlon = False
        use_UTM = True
        assert target_epsg is not None, 'for UTM a EPSG is required'
        lat = None
        lon = None
        utm_x = x
        utm_y = y

    if use_latlon:
        zone = get_UTM_zone_from_longitude(lon)
        target_epsg = get_EPSG_for_UTM(latitude=lat, zone=zone)
        utm_xy = convert_latlon_to_utm(lat=lat, lon=lon, trg_epsg=target_epsg)
        lat = None
        lon = None
        utm_x = utm_xy[0]
        utm_y = utm_xy[1]
        use_latlon = False
        use_UTM = True

    available_sensors = ['S2', 'L8', 'L7', 'L5']
    collections = ['COPERNICUS/S2', 'LANDSAT/LC08/C01/T1_TOA', 'LANDSAT/LE07/C01/T1_TOA','LANDSAT/LT05/C01/T1_TOA']

    if use_HOT_Mask is True:
        if 'red' not in rgb_combination:
            rgb_combination.append('red')
            # rgb_combination.insert(0,'red')
        if 'blue' not in rgb_combination:
            rgb_combination.append('blue')
            # rgb_combination.insert(0,'blue')

    image_chips = []
    profile_results = []
    rgb_initial = None

    for iCol, sensor in enumerate(available_sensors):

        if sensor in sensors:

            # in case QA has been added previously to rgb_comb, remove it
            if 'QA' in rgb_combination:
                ipop = rgb_combination.index('QA')
                rgb_combination.pop(ipop)

            if rgb_initial is not None: rgb_combination = list(rgb_initial)

            # ensure band 9 for new s2 cloudmasking
            if sensor == 'S2':
                if rgb_initial is None: rgb_initial = list(rgb_combination) # make a safe copy
                if 'wv' not in rgb_combination: rgb_combination.append('wv')
                if 'swir1' not in rgb_combination: rgb_combination.append('swir1')
                if 'nir' not in rgb_combination: rgb_combination.append('nir')
                if 'red' not in rgb_combination: rgb_combination.append('red')
                if 'blue' not in rgb_combination: rgb_combination.append('blue')
            else:
                if 'wv' in rgb_combination:
                    # for other sensors make sure to remove wv
                    ipop = rgb_combination.index('wv')
                    rgb_combination.pop(ipop)


            if not profile_only:

                gee_band_ids = map_RGB_to_bands(rgb_combination, sensor)

                ChipData = eeRetrieval(collection=collections[iCol],
                                       maxCloud=maxCloud,
                                       startDate=start_date,
                                       endDate=end_date,
                                       lat=lat, lon=lon,
                                       utm_x=utm_x, utm_y=utm_y,
                                       epsg=target_epsg,
                                       rectangle=True,rectDist=windowSize,
                                       pixel_size = pixel_size,
                                       bandList=gee_band_ids,
                                       bandNames=rgb_combination,
                                       )

                ChipData.define_geometry()
                ChipData.get_data()

                try:
                    if len(ChipData.retrieval) > 1:
                        chips = ExtractImageChipsFromEERetrieval(ChipData, nb=len(rgb_combination), rgb_combination=rgb_combination,
                                                             use_HOT_Mask=True, HOT_crit=hot_crit, HOT_cutoff=HOT_cutoff, verbose=False)
                        image_chips.extend(chips)
                except:
                    if ChipData.retrieval is None:
                        continue

                    elif ChipData.retrieval is False:
                        monthly_ranges = get_monthly_ranges(from_date=start_date, to_date=end_date)
                        for mrange in monthly_ranges:

                            if check_sensor_obs_period(mrange[1], sensor) is False:
                                continue

                            if True:
                                print 'retrieving {}-{} for {}'.format(mrange[0],mrange[1],sensor)

                            m_start_data = mrange[0]
                            m_end_data = mrange[1]
                            ChipData = eeRetrieval(collection=collections[iCol],
                                                   maxCloud=maxCloud,
                                                   startDate=m_start_data,
                                                   endDate=m_end_data,
                                                   lat=lat, lon=lon,
                                                   utm_x=utm_x, utm_y=utm_y,
                                                   epsg=target_epsg,
                                                   rectangle=True, rectDist=windowSize,
                                                   pixel_size=pixel_size,
                                                   bandList=gee_band_ids,
                                                   bandNames=rgb_combination,
                                                   )

                            ChipData.define_geometry()
                            ChipData.get_data()

                            if ((ChipData.retrieval is None) or (ChipData.retrieval is False)):
                                continue

                            chips = ExtractImageChipsFromEERetrieval(ChipData, nb=len(rgb_combination),rgb_combination=rgb_combination,
                                                                             use_HOT_Mask=True, HOT_crit=hot_crit, HOT_cutoff=HOT_cutoff, verbose=False)
                            image_chips.extend(chips)

            if profile_mean is True:
                rectDist = profile_window*pixel_size
                rectangle = True
            else:
                rectDist = None
                rectangle = None

            # todo: currently defaults to NDVI
            gee_ndvi_band_ids = map_RGB_to_bands(['blue','red','nir'], sensor)

            if ((sensor != 'S2')and(pixel_size == 10)):
                pixel_size = 20

            profile = eeRetrieval(collection=collections[iCol],
                                  maxCloud=maxCloud,pixel_size=pixel_size,
                                  startDate=start_date,endDate=end_date,
                                  lat=lat, lon=lon, utm_x=utm_x, utm_y=utm_y,
                                  rectangle=rectangle, rectDist=rectDist,
                                  bandList=gee_ndvi_band_ids, bandNames=['blue','red','nir'],
                                  epsg=target_epsg)

            profile.define_geometry()
            profile.get_data()
            profile = None if profile.retrieval is False else profile
            profile_results.append(profile)

        else:
            profile_results.append(None)

    if profile_mean is True:
        pixel_timeseries = ExtractAverageTimeSeriesProfileFromEEretrieval(profile_results,
                                                                          rgb_combination=rgb_combination,
                                                                          POI_ID=POI_ID)
    else:
        pixel_timeseries = TimeSeriesProfileFromEERetrieval(S2=profile_results[0],
                                                            L8=profile_results[1],
                                                            L7=profile_results[2],
                                                            L5=profile_results[3],
                                                            POI_ID=POI_ID)

    image_chips_sorted = sort_imageChips(image_chips)

    if overplot_modis:

        modis_colls = ['MODIS/006/MOD09Q1', 'MODIS/006/MYD09Q1']
        modis_doy_colls = ['MODIS/006/MOD09A1', 'MODIS/006/MYD09A1']
        modis_results = []
        modis_DOYs = []

        for iCol in range(len(modis_colls)):
            # for modis maxCloud is ignored...
            modis = eeRetrieval(collection=modis_colls[iCol],
                                startDate=start_date, endDate=end_date,
                                lat=lat, lon=lon,utm_x=utm_x, utm_y=utm_y,
                                pixel_size=250, bandList=['sur_refl_b01', 'sur_refl_b02'],
                                bandNames=['red', 'nir'], epsg=target_epsg)

            DOYs = eeRetrieval(collection=modis_doy_colls[iCol],
                               startDate=start_date, endDate=end_date,
                               lat=lat, lon=lon, utm_x=utm_x, utm_y=utm_y,pixel_size=250,
                               bandList=['DayOfYear'], bandNames=['DOY'], epsg=target_epsg)

            modis.define_geometry()
            modis.get_data()

            DOYs.define_geometry()
            DOYs.get_data()

            modis_results.append(modis)
            modis_DOYs.append(DOYs)

        modis_timeseries = ModisTimeSeriesProfileFromEERetrieval(modis_terra=modis_results[0],terra_doys=modis_DOYs[0],
                                                                 modis_aqua=modis_results[1],aqua_doys=modis_DOYs[1])
    else:
        modis_timeseries = None

    print('time for data retrieval: {0}'.format(time.time() - start_time_total))
    plot_time_total = time.time()

    if csv_only is False:
        fig = plotEEtimeSeriesAndImageChips(plotProfile=True,
                                            plotImageChips=not(profile_only),
                                            timeSeriesProfile=pixel_timeseries,
                                            imageChips=image_chips_sorted,
                                            monthly_chips=monthly_chips,
                                            filtered_profile=filtered_profile,
                                            modis_profile = modis_timeseries,
                                            hot_crit=hot_crit, outdir=outdir)

    if csv_output: #export_csv

        try:
            fig
        except NameError:
            fig = '{0}_{1}-{2}.csv'.format(pixel_timeseries.POI_ID, pixel_timeseries.startDate, pixel_timeseries.endDate)

        out_csv = os.path.join(outdir, os.path.basename(fig)[:-4]+'_values.csv')

        np.savetxt(out_csv,np.column_stack((pixel_timeseries.date_time.astype(str),
                                           np.array(pixel_timeseries.date_doy).astype(str),
                                           pixel_timeseries.ObsTimeSeries.astype(str),
                                           pixel_timeseries.sensor.astype(str))),
                                           newline='\r', delimiter=',', fmt='%s')

    print('time for plotting: {0}'.format(time.time() - plot_time_total))

    return fig


def compute_bits_gee(image, start, end, newName):
    """ Compute the bits of an image

    :param image: image that contains the band with the bit information
    :type image: ee.Image
    :param start: start bit
    :type start: int
    :param end: end bit
    :type end: int
    :param newName: new name for the band
    :type newName: str
    :return: a single band image of the extracted bits, giving the band
        a new name
    :rtype: ee.Image
    """
    pattern = 0

    for i in range(start, end + 1):
        pattern += 2 ** i

    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)

########################################################################################################################
##########      Below masking functions use the QA layer for Landsat and S2 ############################################
##########      For Landsat these QA bit are quite useful                   ############################################
##########      For S2 the user defined making function should be used: s2_cloudmasking() ##############################
########################################################################################################################

def S2_TOA_masking(img):
    bandQA = img.select('QA60')
    # bit_cloud = ee.Number(2).pow(10).int()
    # bit_cirrs = ee.Number(2).pow(11).int()
    # mask = bandQA.bitwiseAnd(bit_cloud).eq(0).and(bandQA.bitwiseAnd(bit_cirrs).eq(0))

    opaque = compute_bits_gee(bandQA, 10, 10, "opaque")
    cirrus = compute_bits_gee(bandQA, 11, 11, "cirrus")
    mask = opaque.Or(cirrus)
    result = img.updateMask(mask.Not())
    return result


def L8_TOA_masking(img):
    bandQA = img.select('BQA')

    satrd = compute_bits_gee(bandQA, 2, 3, "satrd")
    cloud = compute_bits_gee(bandQA, 4, 4, "cloud")
    shadow = compute_bits_gee(bandQA, 7, 8, "shadow")
    snow = compute_bits_gee(bandQA, 9, 10, "snow")
    cirrus = compute_bits_gee(bandQA, 11, 12, "cirrus")

    mask = cloud.eq(1).And(shadow.eq(1)).And(cirrus.eq(1))
    result = img.updateMask(mask.Not())

    # https://gis.stackexchange.com/questions/274048/apply-cloud-mask-to-landsat-imagery-in-google-earth-engine-python-api?rq=1
    # result = img.updateMask(cloud.eq(0)).updateMask(shadow.eq(0))#.updateMask(cirrus.eq(0))

    return result


def L7_TOA_masking(img):
    bandQA = img.select('BQA')

    satrd = compute_bits_gee(bandQA, 2, 3, "satrd")
    cloud = compute_bits_gee(bandQA, 4, 4, "cloud")
    shadow = compute_bits_gee(bandQA, 7, 8, "shadow")
    snow = compute_bits_gee(bandQA, 9, 10, "snow")

    mask = cloud.eq(1).And(shadow.eq(1)).And(snow.eq(1))
    result = img.updateMask(mask.Not())

    return result


class eeRetrieval:

    ee.Initialize()

    def __init__(self, collection=None, sensor=None, maxCloud=None, startDate=None, endDate=None,
                 lat=None, lon=None, utm_x=None, utm_y=None, utm_ul_xy=None, utm_lr_xy=None, latlon_ul_xy=None, latlon_lr_xy=None,
                 bandList=None, bandNames=None, centerPOI=None, epsg=None, rectangle=None, rectDist=None, pixel_size = None, eeGeometry=None,
                 eeImageCollection=None, retrieval=None):
        '''
        :param collection: str, GEE collection specification, e.g. 'COPERNICUS/S2'
        :param maxCloud:
        :param startDate: str 'YYYY-MM-DD'
        :param endDate: str 'YYYY-MM-DD'
        :param lat:
        :param lon:
        :param utm_x:
        :param utm_y:
        :param bandList:
        :param centerPOI:

        '''

        use_QA_Mask = True

        if epsg is None:
            epsg = 4326

        self.collection = collection
        self.sensor = sensor
        self.maxCloud = maxCloud
        self.startDate = startDate
        self.endDate = endDate
        self.lat = lat
        self.lon = lon
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.utm_ul_xy = utm_ul_xy
        self.utm_lr_xy = utm_lr_xy
        self.latlon_ul_xy = latlon_ul_xy
        self.latlon_lr_xy = latlon_lr_xy
        self.bandList = bandList
        self.bandNames = bandNames
        self.centerPOI = centerPOI
        self.epsg = epsg
        self.rectangle = rectangle
        self.rectDist = rectDist
        self.pixel_size = pixel_size

        self.eeGeometry = eeGeometry
        self.eeImageCollection = eeImageCollection
        self.retrieval = retrieval

        all_collections = ['COPERNICUS/S1_GRD',  # Oct 3, 2014 - Jan 25, 2018
                           'COPERNICUS/S2',  # Jun 23, 2015 - Jan 23, 2018
                           'LANDSAT/LC08/C01/T1_TOA',
                           'LANDSAT/LE07/C01/T1_TOA',
                           'LANDSAT/LT05/C01/T1_TOA',
                           'LANDSAT/LC8_L1T_TOA_FMASK',
                           'LANDSAT/LT5_L1T_TOA_FMASK',
                           'LANDSAT/LT4_L1T_TOA_FMASK',
                           'MODIS/006/MOD13Q1','MODIS/006/MYD13Q1',
                           'MODIS/006/MOD09Q1','MODIS/006/MYD09Q1',
                           'MODIS/006/MOD09A1','MODIS/006/MYD09A1',
                           ]

        S1_bands = ['HH', 'HV', 'VV', 'VH', 'angle']
        S2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8a', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60']

        assert epsg is not None, 'epsg needs to be defined'

        if (lat and lon) is not None:
            assert (utm_x and utm_y) is None, 'Provide either lat/lon or UTM x/y, not both'
            ptx = lon
            pty = lat

        if (utm_x and utm_y) is not None:
            assert (lat and lon) is None, 'Provide either lat/lon or UTM x/y, not both'
            assert epsg != 4326, 'provide CRS/EPSG for UTM'
            ptx = utm_x
            pty = utm_y

        if rectangle:
            assert rectDist is not None, "to retrieve a rectangle a rectangle distance has to be set [m]"
            #assert rectDist % pixel_size == 0, "rectangle size needs to be a multiple of the pixel size"
            if rectDist % pixel_size != 0:
                self.rectDist = rectDist + (rectDist % pixel_size)


        assert collection in all_collections, \
            "collection not in list of known collections: {0}".format(all_collections)

        if 'LC08' in collection:
            self.sensor = "L8"
        elif 'LE07' in collection:
            self.sensor = "L7"
        elif 'S2' in collection:
            self.sensor = "S2"
        elif 'S1_GRD' in collection:
            self.sensor = "S1"
        elif 'MODIS' in collection:
            self.sensor = "MODIS"
        elif 'LT05' in collection:
            self.sensor = "L5"
        else:
            raise NotImplementedError, "other collections "

        if bandList:
            # todo PG set to TRUE and explore S2 TOA masking
            if use_QA_Mask: # experiment with TOA cloud masks #TODO PG
                if self.sensor == "S2":
                    if 'QA60' not in bandList:
                        bandList.append('QA60')
                        bandNames.append('QA')
                elif self.sensor == "L8":
                    if 'BQA' not in bandList:
                        bandList.append('BQA')
                        bandNames.append('QA')
                elif self.sensor == "L7":
                    if 'BQA' not in bandList:
                        bandList.append('BQA')
                        bandNames.append('QA')
                else:
                    pass
            if bandNames:
                assert len(bandList) == len(bandNames)

            # if rectangle:
            #     if self.sensor == "S2":
            #         if 'swir1' not in rgb_combination: rgb_combination.append('swir1')
            #         if 'nir' not in rgb_combination: rgb_combination.append('nir')
            #         if 'red' not in rgb_combination: rgb_combination.append('red')
            #         if 'blue' not in rgb_combination: rgb_combination.append('blue')

            assert isinstance(bandList, list)
            bandList = ee.List(bandList)

        assert len(startDate.split('-')) == 3 & len(endDate.split('-')) == 3,\
            'start and end date have to be of form "YYY-MM-DD"'

        ################################################################################################################
        if maxCloud is None: maxCloud = 100

        if collection == 'COPERNICUS/S2':
            if use_QA_Mask:
                if rectangle:
                    # if rectangle area is extracted, we can use shadow matching etc.
                    self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                                filterDate(startDate, endDate). \
                                                filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', maxCloud). \
                                                map(s2_cloudmasking). \
                                                filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
                else:
                    # if not we have to rely on QB bit information for clouds and shadows
                    self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                                filterDate(startDate, endDate). \
                                                filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', maxCloud). \
                                                map(S2_TOA_masking). \
                                                filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
            else:
                self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                            filterDate(startDate, endDate). \
                                            filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', maxCloud). \
                                            filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        elif collection == 'LANDSAT/LC08/C01/T1_TOA':
            if use_QA_Mask:
                self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                            filterDate(startDate, endDate). \
                                            filter(ee.Filter.lt('CLOUD_COVER', maxCloud)). \
                                            map(L8_TOA_masking). \
                                            filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
            else:
                self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                            filterDate(startDate, endDate). \
                                            filter(ee.Filter.lt('CLOUD_COVER', maxCloud)). \
                                            filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        elif collection == 'LANDSAT/LE07/C01/T1_TOA':
            if use_QA_Mask:
                self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                            filterDate(startDate, endDate). \
                                            filter(ee.Filter.lt('CLOUD_COVER', maxCloud)). \
                                            map(L7_TOA_masking). \
                                            filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
            else:
                self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                            filterDate(startDate, endDate). \
                                            filter(ee.Filter.lt('CLOUD_COVER', maxCloud)). \
                                            filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        else:
            self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
                                        filterDate(startDate, endDate). \
                                        filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
                                        # filter(ee.Filter.lt('CLOUD_COVER', maxCloud)). \

        ################################################################################################################

        # if maxCloud:
        #
        #     if collection == 'COPERNICUS/S2':
        #
        #         if not use_QA_Mask:
        #
        #             self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList).\
        #                                         filterDate(startDate, endDate).\
        #                                         filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', maxCloud).\
        #                                         filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        #
        #         if use_QA_Mask: # todo PG set to TRUE and explore S2 TOA masking
        #
        #             self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList). \
        #                                                         filterDate(startDate, endDate). \
        #                                                         filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', maxCloud). \
        #                                                         map(S2_TOA_masking).\
        #                                                         filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        #     else:
        #         self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList).\
        #                                     filterDate(startDate,endDate).\
        #                                     filter(ee.Filter.lt('CLOUD_COVER', maxCloud)).\
        #                                     filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))
        # else:
        #     self.eeImageCollection = ee.ImageCollection(self.collection).select(bandList).\
        #                                 filterDate(startDate, endDate).filterBounds(ee.Geometry.Point([ptx, pty], 'epsg:{}'.format(epsg)))

    def query_bands(self, sensor):
        S1_bands = ['HH', 'HV', 'VV', 'VH', 'angle']
        S2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8a', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60']
        L8_bands = []
        L7_bands = []

        if sensor == 'S2':
            return S2_bands
        elif sensor == 'S1':
            return S1_bands
        elif sensor == 'L8':
            L8_bands
        elif sensor == 'L7':
            L7_bands


    def define_geometry(self):

        # define point:
        if self.rectangle is None:

            # use point geometry
            if ((self.utm_x is None) & (self.lon is not None)):
                # use lat lon point todo: if lat/lon then default epsg should be used, or left blank
                is_geodesic = True
                self.eeGeometry = ee.Geometry.Point(list([self.lon, self.lat]), "epsg:{0}".format(4326))

            elif ((self.utm_x is not None) & (self.lon is None)):
                # UTM point
                is_geodesic = False
                self.eeGeometry = ee.Geometry.Point(list([self.utm_x, self.utm_y]), "epsg:{0}".format(self.epsg))


        elif self.rectangle is True:

            # use rectangle geometry
            if (((self.utm_x  and self.utm_y) is None) & ((self.lon  and self.lat) is not None)):

                # use lat lon coords #todo ensure center pixel for crosshair
                #pad = (((self.rectDist + self.pixel_size) / 2.) / 1000.) * 0.008983
                pad = ((self.rectDist/2.) / 1000.) * 0.008983

                self.eeGeometry = ee.Geometry.Rectangle([self.lon - pad, self.lat - pad, self.lon + pad, self.lat + pad])

                self.latlon_ul_xy = [self.lon - pad, self.lat + pad]
                self.latlon_lr_xy = [self.lon + pad, self.lat - pad]

            elif (((self.utm_x and self.utm_y) is not None) & ((self.lon and self.lat) is None)):

                is_geodesic = False

                ULXY = offset_utm_coord(x=self.utm_x, y=self.utm_y, dist=-1 * (self.rectDist / 2.), ps=self.pixel_size)
                LRXY = offset_utm_coord(x=self.utm_x, y=self.utm_y, dist=self.rectDist / 2., ps=self.pixel_size)

                # below adds half a pixel size to the rectangle if rect size is a multiple of pixel size - no used anymore
                # hps = int(self.pixel_size/2.) if self.rectDist % self.pixel_size == 0 else 0 # one half pixel size
                # ULXY = offset_utm_coord(x=self.utm_x - hps, y=self.utm_y + hps, dist=-1 * (self.rectDist / 2.), ps=self.pixel_size)
                # LRXY = offset_utm_coord(x=self.utm_x + hps, y=self.utm_y - hps, dist=self.rectDist / 2., ps=self.pixel_size)
                # self.rectDist = self.rectDist+(2*hps)

                if False:
                    print 'ULXY = {0},{1}'.format(ULXY[0],ULXY[1])
                    print 'LRXY = {0},{1}'.format(LRXY[0], LRXY[1])

                    print 'X distance: {0}'.format(LRXY[0] - ULXY[0])
                    print 'Y distance: {0}'.format(ULXY[1] - LRXY[1])

                self.utm_ul_xy = ULXY
                self.utm_lr_xy = LRXY

                LLXY = [ULXY[0], LRXY[1]]
                URXY = [LRXY[0], ULXY[1]]

                self.eeGeometry = ee.Geometry.Polygon(geodesic=False, coords=[LLXY, LRXY, URXY, ULXY, ULXY], proj="epsg:{0}".format(self.epsg))

            else:
                raise Exception, "something is weired (lat/lon vs. utm x/y)"

    def get_data(self):
        '''

        :return: returns False if GEE request exceeds limit
                 returns None if not data is found for requested time and location
        '''

        try:
            self.retrieval = self.eeImageCollection.getRegion(self.eeGeometry, self.pixel_size, "epsg:{0}".format(self.epsg)).getInfo()

            print "{0} - Number of observations retrieved: {1} ({2} - {3})"\
                .format(self.collection, len(np.unique(np.array([d[0] for d in self.retrieval[1:]]))), self.startDate, self.endDate)

        except Exception, e:
            print 'Earth Engine error during retrieval attempt: \n --> {0}'.format(e)
            self.retrieval = False
        except:
            print 'no images in collections for time period: {0} - {1}'.format(self.startDate, self.endDate)
            self.retrieval = None

    def clean_data(self):
        pass

    def date_time(self):
        pass


class eeImageChip:

    def __init__(self, image_chip=None, acq_id=None, sensor=None, date_time=None, header=None, pixel_size=None, pct_clear=None, rgb_combination=None):
        self.image_chip = image_chip
        self.acq_id = acq_id
        self.sensor = sensor
        self.date_time = date_time
        self.header = header
        self.pixel_size = pixel_size
        self.pct_clear = pct_clear
        self.rgb_combination = rgb_combination

class eePixelTimeSeries:

    def __init__(self, ObsTimeSeries=None, ObsType=None, acq_id=None, sensor=None, date_time=None, date_doy=None, startDate=None, endDate=None,
                 utm_x=None, utm_y=None, lat=None, lon=None, pixel_size=None, ObsHOT=None, ObsMask=None, ObsMaskType=None, POI_ID=None,
                 is_mean=False,mean_window=None):

        self.ObsTimeSeries = ObsTimeSeries
        self.ObsType = ObsType
        self.acq_id = acq_id
        self.sensor = sensor
        self.date_time = date_time
        self.date_doy = date_doy
        self.startDate = startDate
        self.endDate = endDate
        self.pixel_size = pixel_size
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.lat = lat
        self.lon = lon
        self.ObsHOT = ObsHOT
        self.ObsMask = ObsMask
        self.ObsMaskType = ObsMaskType
        self.POI_ID = POI_ID
        self.is_mean = is_mean
        self.mean_window = mean_window

def ExtractAverageTimeSeriesProfileFromEEretrieval(dataEEcollections, hot_crit=0.02, rgb_combination=None, POI_ID=None):

    meanObsValues = list()
    meanHOTobs = list()
    all_time_stamps = []
    all_obs_sensor = []
    all_acq_ids = []

    collected = 0

    for eeData in dataEEcollections:

        if eeData is None:
            continue

        if collected == 0:
            # grab common attributes
            startDate = eeData.startDate
            endDate = eeData.endDate
            lat = eeData.lat
            lon = eeData.lon
            utm_x = eeData.utm_x
            utm_y = eeData.utm_y
            pixel_size = eeData.pixel_size
            collected = 1

        nb = len(eeData.bandList)
        if eeData.sensor == "S2":
            scale = 10000.
        else:
            scale = 1.

        retrievalData = eeData.retrieval

        if isinstance(retrievalData, bool):
            print ""


        acqs = np.unique(np.array([d[0] for d in retrievalData[1:]]))
        IDsArr = np.array([d[0] for d in retrievalData[1:]])
        acqTimeArr = np.array([d[3] for d in retrievalData[1:]])
        retrievalArr = np.array([d[4:] for d in retrievalData[1:]])

        # try:
        #     if rgb_combination is not None:
        #         i = 0 if len(rgb_combination) == 3 else 1
        #         iRGB1 = eeData.bandNames.index(rgb_combination[i + 0])
        #         iRGB2 = eeData.bandNames.index(rgb_combination[i + 1])
        #         iRGB3 = eeData.bandNames.index(rgb_combination[i + 2])
        #         if use_HOT_Mask:
        #             i_blue = retrieval.bandNames.index('blue')
        #             i_red = retrieval.bandNames.index('red')
        try:
            i_blue = eeData.bandNames.index('blue')
            i_red = eeData.bandNames.index('red')
            i_nir = eeData.bandNames.index('nir')
            #i_swir1 = eeData.bandNames.index('swir1')
            #i_swir2 = eeData.bandNames.index('swir2')
        except:
            raise Exception

        ns = eeData.rectDist / eeData.pixel_size
        nl = eeData.rectDist / eeData.pixel_size

        for id in acqs:

            inds = np.where(IDsArr == id)

            i_time_stamp = np.datetime64(int(np.min(acqTimeArr[inds])), 'ms')
            if i_time_stamp in all_time_stamps:
                continue
            all_acq_ids.append(IDsArr[inds][0])
            all_obs_sensor.append(eeData.sensor)
            all_time_stamps.append(i_time_stamp)

            acqData = retrievalArr[inds]

            try:
                acqArr = acqData.reshape(ns, nl, nb).astype(np.float32)
            except:
                raise Exception, 'failed to reconstruct array'

            if False:
                zero_msk = np.where((acqArr[:, :, i_blue] == 0.0) & (acqArr[:,:,i_red] == 0.0))
                hotArr = (acqArr[:, :, i_blue] - 0.5 * acqArr[:,:,i_red] - 0.08)
                hotArr[zero_msk] = 0.00
                hot_ok = np.where((hotArr <= hot_crit) & (hotArr != 0.000))
                meanHOTobs.append(hotArr[hot_ok])

            nNIR = np.mean(acqArr[:,:,i_nir])/scale
            mRED = np.mean(acqArr[:,:,i_red])/scale
            mBLU = np.mean(acqArr[:, :, i_blue])/scale

            meanObs = ((nNIR - mRED)/(nNIR + mRED))
            if np.isnan(meanObs):
                meanObsValues.append(0.0)
            else:
                meanObsValues.append(meanObs)

            meanHOT = (mBLU - 0.5 * mRED - 0.08)
            if np.isnan(meanHOT):
                meanHOTobs.append(0.00)
            else:
                meanHOTobs.append(meanHOT)


    time_arr = np.array(all_time_stamps)
    iTime = time_arr.argsort()

    obs_arr_sorted = np.array(meanObsValues)[iTime]
    HOTsorted = np.array(meanHOTobs)[iTime]
    sensor_id_arr_sorted = np.array(all_obs_sensor)[iTime]
    acq_ids_sorted = np.array(all_acq_ids)[iTime]
    time_arr_sorted = time_arr[iTime]

    doys = datetime2doy(time_arr_sorted)

    profile = eePixelTimeSeries(ObsTimeSeries=obs_arr_sorted,
                                ObsType='NDVI_MEAN',
                                acq_id=acq_ids_sorted,
                                sensor=sensor_id_arr_sorted,
                                date_time=time_arr_sorted,
                                date_doy = doys,
                                startDate=startDate,
                                endDate=endDate,
                                ObsHOT=HOTsorted,
                                ObsMask=np.where((HOTsorted <= hot_crit)&(HOTsorted != 0.00)),
                                ObsMaskType='hot ok',
                                utm_x=utm_x,
                                utm_y=utm_y,
                                lat=lat,
                                lon=lon,
                                pixel_size=pixel_size,
                                POI_ID=POI_ID,
                                is_mean=True,
                                mean_window=ns)
    return profile

def ExtractImageChipsFromEERetrieval(retrieval, nb=None, rgb_combination=None, use_HOT_Mask=False,
                                     HOT_crit=0.02, HOT_cutoff=50,verbose=False, render_chip=False, outdir=None):
    '''

    :param retrieval: retrieval from GEE
    :param nb: number of bands per retrieved sensor observation
    :param use_HOT_Mask: use Haze Optimized Transformation as a proxy for cloud cover
    :paramm HOT_crit: HOT values above HOT_crit are considered clouded
    :param HOT_cutoff: if a image chip has a HOT based cloud cover estimate above this value, skip this chip
    :param verbose: print assessed cloud coverage per chip
    :return: eeImageChip object
    '''

    if retrieval.retrieval is False:
        return

    assert nb is not None

    if render_chip:
        count = 0
        fps_chips = []

      # HOT threshold, for SR use 0.00 for TOA use 0.02 or higher

    retrievalData = retrieval.retrieval

    acqs = np.unique(np.array([d[0] for d in retrievalData[1:]]))
    IDsArr = np.array([d[0] for d in retrievalData[1:]])
    acqTimeArr = np.array([d[3] for d in retrievalData[1:]])
    retrievalArr = np.array([d[4:] for d in retrievalData[1:]])

    if False:
        lonArr = np.array([d[1] for d in retrievalData[1:]])
        latArr = np.array([d[2] for d in retrievalData[1:]])
        #latArr[inds]
        #lonArr[inds]

    #np.unique(np.array([d[0] for d in retrievalData[1:]]))
    if rgb_combination is not None:
        i = 0 #if len(rgb_combination) == 3 else 1
        iRGB1 = retrieval.bandNames.index(rgb_combination[i+0])
        iRGB2 = retrieval.bandNames.index(rgb_combination[i+1])
        iRGB3 = retrieval.bandNames.index(rgb_combination[i+2])
        if use_HOT_Mask:
            i_blue = retrieval.bandNames.index('blue')
            i_red = retrieval.bandNames.index('red')
    else:
        i_blue = retrieval.bandNames.index('blue')
        i_red = retrieval.bandNames.index('red')
        i_nir = retrieval.bandNames.index('nir')
        i_swir1 = retrieval.bandNames.index('swir1')
        try:
            i_swir2 = retrieval.bandNames.index('swir2')
        except:
            pass
            #print "no swir2 band provided"

    chipObjects = list()
    msg = False
    for id in acqs:  # [::-1]:

        inds = np.where(IDsArr == id)
        acqData = retrievalArr[inds]

        ns = retrieval.rectDist / retrieval.pixel_size
        nl = retrieval.rectDist / retrieval.pixel_size

        if False:
            print 'X distance: {0}'.format(retrieval.utm_lr_xy[0] - retrieval.utm_ul_xy[0])
            print 'Y distance: {0}'.format(retrieval.utm_ul_xy[1] - retrieval.utm_lr_xy[1])

            ns = max(1,int(abs(retrieval.utm_lr_xy[0] - retrieval.utm_ul_xy[0]) / retrieval.pixel_size))
            nl = max(1,int(abs(retrieval.utm_ul_xy[1] - retrieval.utm_lr_xy[1]) / retrieval.pixel_size))

        try:
            #acqArr = acqData.reshape(npx, npx, nb).astype(np.float32)
            acqArr = acqData.reshape(ns, nl, nb).astype(np.float32)
        except:
            raise Exception

        # ensure North up::
        list2stack = []
        for b in range(acqArr.shape[2]):
            list2stack.append(np.flip(acqArr[:, :, b], 0))
            # list2stack.append(np.transpose(acqData[:,b]))
        acqArr = np.dstack(list2stack)

        if use_HOT_Mask:

            if retrieval.sensor == "S2":
                scale_factor = 10000.
            else:
                scale_factor = 1.

            blue = acqArr[:, :, i_blue].astype(np.float32) / scale_factor
            red = acqArr[:, :, i_red].astype(np.float32) / scale_factor

            zero_msk = np.where((blue == 0.0) & (red == 0.0))
            # nan_msk = np.where((np.isnan(blue) == True) & (np.isnan(red) == True))
            #nan_msk = np.where((np.isnan(blue) == True) | (np.isnan(red) == True))

            hot = (blue - 0.5 * red - 0.08)

            hot[zero_msk] = 0.00
            #hot[nan_msk] = 0.00
            hot = np.where(np.isnan(blue),0.0, hot)
            hot = np.where(np.isnan(red), 0.0, hot)
            hot = np.where(np.isnan(hot), 0.0, hot)

            # hot_ok = np.count_nonzero((hot <= HOT_crit) & (hot != 0.000))
            i_hot_ok = np.where((hot <= HOT_crit) & (hot != 0.000))
            hot_ok = i_hot_ok[0].size

            pct_ok = (float(hot_ok) / float(ns * nl)) * 100.
            # pct_ok = (float(hot_ok) / float(npx * npx)) * 100.

            if verbose:
                print('{0}% OF PIXELS OK: {1}'.format(pct_ok, id))
            if pct_ok < HOT_cutoff:
                continue

            acqArr = np.where(np.isnan(acqArr), 0.0, acqArr)

            rgb1_sminmax = np.percentile(acqArr[:, :, iRGB1][i_hot_ok], [2, 98])
            rgb2_sminmax = np.percentile(acqArr[:, :, iRGB2][i_hot_ok], [2, 98])
            rgb3_sminmax = np.percentile(acqArr[:, :, iRGB3][i_hot_ok], [2, 98])

        else:
            acqArr = np.where(np.isnan(acqArr), 0.0, acqArr)
            zero_msk = np.where(acqArr == 0.0)
            rgb1_sminmax = np.percentile(acqArr[:, :, iRGB1][zero_msk], [2, 98])
            rgb2_sminmax = np.percentile(acqArr[:, :, iRGB2][zero_msk], [2, 98])
            rgb3_sminmax = np.percentile(acqArr[:, :, iRGB3][zero_msk], [2, 98])

        # nir_sminmax = np.percentile(acqArr[:, :, i_nir][i_hot_ok], [2, 98])
        # swir_sminmax = np.percentile(acqArr[:, :, i_swir1][i_hot_ok], [2, 98])
        # red_sminmax = np.percentile(acqArr[:, :, i_red][i_hot_ok], [2, 98])

        byte_chip = np.zeros((ns, nl, 3), dtype=np.uint8)

        byte_chip[:, :, 0] = bytescale(acqArr[:, :, iRGB1], cmin=max(0, rgb1_sminmax[0]), cmax=rgb1_sminmax[1])
        byte_chip[:, :, 1] = bytescale(acqArr[:, :, iRGB2], cmin=max(0, rgb2_sminmax[0]), cmax=rgb2_sminmax[1])
        byte_chip[:, :, 2] = bytescale(acqArr[:, :, iRGB3], cmin=max(0, rgb3_sminmax[0]), cmax=rgb3_sminmax[1])

        # for instant redering of individual chips, add keyword render_chip and save PNG file
        if render_chip:

            if retrieval.sensor == 'S2':
                date_str = '{0}_{1}'.format(id[:8], retrieval.sensor)
            else:
                # LC08_194028_20170217_ImageChip
                date_str = '{0}_{1}'.format(id[12:20], retrieval.sensor)


            outname_chip = os.path.join(outdir, "{0}_{1}.png".format(date_str, "ImageChip"))
            # outname_chip = os.path.join(outdir, "{0}_{1}_{2}.png".format(str(count+1).zfill(3), id, "ImageChip"))

            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(byte_chip, interpolation='nearest')
            plt.tight_layout()
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(outname_chip, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            ax=None
            fig=None
            # img = plt.imshow(byte_chip, interpolation='nearest')
            # plt.axis('off')
            # img.axes.get_xaxis().set_visible(False)
            # img.axes.get_yaxis().set_visible(False)
            # plt.savefig(outname_chip, bbox_inches='tight', pad_inches=0, dpi=150)

            count += 1
            fps_chips.append(outname_chip)
            continue

        chip_date_time = timestamp_to_datetime(acqTimeArr[inds[0][0]])

        if chip_date_time in [cp.date_time for cp in chipObjects]:
            pass
            continue

        chipObjects.append(eeImageChip(image_chip=byte_chip,
                                       acq_id=id,
                                       date_time=chip_date_time,
                                       sensor=retrieval.sensor,
                                       pixel_size = retrieval.pixel_size,
                                       pct_clear=pct_ok,
                                       rgb_combination=rgb_combination))

    if verbose:
        print '{0} out of {1} available image chips passed the HOT based cloud screening (hot_cutoff={2}, hot_crit={3})...'.\
            format(len(chipObjects),len(acqs),HOT_cutoff, HOT_crit)

    if render_chip:
        return fps_chips

    return chipObjects

def ModisTimeSeriesProfileFromEERetrieval(modis_terra=None, modis_aqua=None,
                                          aqua_doys=None, terra_doys=None, POI_ID=None):

    all_time_stamps = []
    all_obs_vals = []

    if modis_terra:
        data_terra = clean_EE_obs_data(modis_terra.retrieval)
        terra_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_terra[1:]]
        all_time_stamps.extend(terra_time_stamps)
        all_obs_vals.extend(data_terra[1:])
        i_modis_nir = modis_terra.bandNames.index('nir')
        i_modis_red = modis_terra.bandNames.index('red')
        if terra_doys:
            all_obs_doys = []
            terra_doys = clean_EE_obs_data(terra_doys.retrieval)
            terra_doys = [d[4] for d in terra_doys[1:]]
            all_obs_doys.extend(terra_doys)

    if modis_aqua:
        data_aqua = clean_EE_obs_data(modis_aqua.retrieval)
        aqua_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_aqua[1:]]
        if modis_terra is not None:
            # as the obs doy is not available, for now shift aqua obs time stamp by 4 days
            aqua_time_stamps = aqua_time_stamps + np.timedelta64(3, 'D')
        all_time_stamps.extend(aqua_time_stamps)
        all_obs_vals.extend(data_aqua[1:])
        # just overwrite to make sure we have it
        i_modis_nir = modis_aqua.bandNames.index('nir')
        i_modis_red = modis_aqua.bandNames.index('red')
        if aqua_doys:
            aqua_doys = clean_EE_obs_data(aqua_doys.retrieval)
            aqua_doys = [d[4] for d in aqua_doys[1:]]
            all_obs_doys.extend(aqua_doys)

    startDate = max(np.datetime64(modis_terra.startDate), np.datetime64(modis_aqua.startDate))
    endDate = max(np.datetime64(modis_terra.endDate), np.datetime64(modis_aqua.endDate))

    lat = max([modis_terra.lat, modis_aqua.lat])
    lon = max([modis_terra.lon, modis_aqua.lon])

    utm_x = max([modis_terra.utm_x, modis_aqua.utm_x])
    utm_y = max([modis_terra.utm_y, modis_aqua.utm_y])

    pixel_size = max([modis_terra.pixel_size, modis_aqua.pixel_size])

    time_arr = np.array(all_time_stamps)
    obs_arr = np.array(all_obs_vals)

    # sort the data
    inds = time_arr.argsort()

    obs_arr_sorted = obs_arr[inds]
    time_arr_sorted = time_arr[inds]
    # sensor_id_arr_sorted = sensor_id_arr[inds]
    acq_ids_sorted = [d[0] for d in obs_arr_sorted]

    if terra_doys and aqua_doys:
        #use MODIS obs DOY for as time stamp and shift the aqua time stamp by 12 hours
        years = [int(np.datetime_as_string(y, "Y")) for y in time_arr]
        terra_aqua = np.append(np.repeat(1, len(data_terra[1:])), np.repeat(2, len(data_aqua[1:])))
        doy2timestamp = [np.datetime64('{:04d}-01-01'.format(years[iy]), 'ms') + np.timedelta64(int(doy) - 1, 'D') for iy, doy in enumerate(all_obs_doys)]
        doy2timestamp = np.array([d + np.timedelta64(12, 'h') if terra_aqua[i] == 2 else d for i, d in enumerate(doy2timestamp)])
        doy2timestamp_sorted = doy2timestamp[inds]
        time_stamps = doy2timestamp_sorted
    else:
        time_stamps = time_arr_sorted



    nir = np.array([])
    red = np.array([])

    print('extracting band values for {0} observations...'.format(len(obs_arr_sorted)))
    for vctr in obs_arr_sorted:
        nir = np.append(nir, float(vctr[4+i_modis_nir]) / 10000.)
        red = np.append(red, float(vctr[4+i_modis_red]) / 10000.)

    ndvi = ((nir - red) / (nir + red))
    np.nan_to_num(ndvi, copy=False)

    profile = eePixelTimeSeries(ObsTimeSeries = ndvi,
                                ObsType = 'NDVI',
                                acq_id = acq_ids_sorted,
                                sensor = "MODIS",
                                date_time = time_stamps,
                                startDate = startDate,
                                endDate = endDate,
                                utm_x = utm_x,
                                utm_y = utm_y,
                                lat=lat,
                                lon=lon,
                                pixel_size=pixel_size,
                                POI_ID = POI_ID)
    return profile

def TimeSeriesProfileFromEERetrieval(S2=None, L8=None, L7=None, S1=None, L5=None, POI_ID=None):
    # todo rewrite similar to average profile extraction
    import fnmatch

    hot_crit = 0.02  # HOT threshold, for SR use 0.00 for TOA use 0.02 or higher

    all_time_stamps = []
    all_obs_vals = []

    data_for_common_attributes = None

    if S1:
        data_S1 = S1.retrieval
        S1_time_stamps = np.array([np.datetime64(ts[3], 'ms') for ts in data_S1[1:]])
        startDate = np.datetime64(S1.startDate)
        endDate = np.datetime64(S1.endDate)
    if S2:
        data_S2 = clean_EE_obs_data(S2.retrieval)
        S2_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_S2[1:]]
        ## account for cases within the S2 tile overlap area to only use the unique observations
        if len(np.unique(S2_time_stamps)) != len(S2_time_stamps):
            print('Sentinel2 tile overlap area encountered: reducing {0} observations to {1} unique acquisitions'.
                  format(len(S2_time_stamps),len(np.unique(S2_time_stamps))))
            # get indices for each acquisition ID
            inds_raw = [np.where(s == S2_time_stamps) for s in np.unique(S2_time_stamps)]
            # if an index for a acq ID is of length two, only use the first one
            inds = np.array([i[0][0] if len(i) > 1 else i[0][0] for i in inds_raw])
            S2_time_stamps = list(np.array(S2_time_stamps)[inds])
            uniqS2data = []
            data_S2_copy = np.array(data_S2[1:], copy=True)
            for i in inds:
                uniqS2data.append(list(data_S2_copy[i]))
            uniqS2data.insert(0, data_S2[0])
            data_S2 = uniqS2data
        all_time_stamps.extend(S2_time_stamps)
        all_obs_vals.extend(data_S2[1:])
        i_s2_nir = S2.bandNames.index('nir')
        i_s2_red = S2.bandNames.index('red')
        i_s2_blue = S2.bandNames.index('blue')
        if data_for_common_attributes is None:
            data_for_common_attributes = S2
    if L8:
        data_L8 = clean_EE_obs_data(L8.retrieval)
        L8_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_L8[1:]]
        all_time_stamps.extend(L8_time_stamps)
        all_obs_vals.extend(data_L8[1:])
        i_l8_nir = L8.bandNames.index('nir')
        i_l8_red = L8.bandNames.index('red')
        i_l8_blue = L8.bandNames.index('blue')
        if data_for_common_attributes is None:
            data_for_common_attributes = L8
    if L7:
        data_L7 = clean_EE_obs_data(L7.retrieval)
        L7_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_L7[1:]]
        all_time_stamps.extend(L7_time_stamps)
        all_obs_vals.extend(data_L7[1:])
        i_l7_nir = L7.bandNames.index('nir')
        i_l7_red = L7.bandNames.index('red')
        i_l7_blue = L7.bandNames.index('blue')
        if data_for_common_attributes is None:
            data_for_common_attributes = L7
    if S1:
        S1_VV_arr = np.array([d[4] for d in data_S1[1:]])
        S1_VH_arr = np.array([d[5] for d in data_S1[1:]])
        if data_for_common_attributes is None:
            data_for_common_attributes = S1
    if L5:
        data_L5 = clean_EE_obs_data(L5.retrieval)
        L5_time_stamps = [np.datetime64(ts[3], 'ms') for ts in data_L5[1:]]
        all_time_stamps.extend(L5_time_stamps)
        all_obs_vals.extend(data_L5[1:])
        i_l5_nir = L5.bandNames.index('nir')
        i_l5_red = L5.bandNames.index('red')
        i_l5_blue = L5.bandNames.index('blue')
        if data_for_common_attributes is None:
            data_for_common_attributes = L5

    # get common attributes
    assert data_for_common_attributes is not None
    startDate = np.datetime64(data_for_common_attributes.startDate)
    endDate = np.datetime64(data_for_common_attributes.endDate)

    lat = data_for_common_attributes.lat
    lon = data_for_common_attributes.lon

    utm_x = data_for_common_attributes.utm_x
    utm_y = data_for_common_attributes.utm_y

    pixel_size = data_for_common_attributes.pixel_size
    data_for_common_attributes = None

    sensor_id_arr = np.array(['L8' if str(i[0]).startswith("LC08") else
                              'L7' if str(i[0]).startswith("LE07") else
                              'L5' if str(i[0]).startswith("LT05") else
                              "S2" for i in all_obs_vals])

    time_arr = np.array(all_time_stamps)
    obs_arr = np.array(all_obs_vals)

    # sort the data
    inds = time_arr.argsort()

    obs_arr_sorted = obs_arr[inds]
    time_arr_sorted = time_arr[inds]
    sensor_id_arr_sorted = sensor_id_arr[inds]
    acq_ids_sorted = [d[0] for d in obs_arr_sorted]

    nir = np.array([])
    red = np.array([])
    blue = np.array([])

    print('extracting band values for {0} observations...'.format(len(obs_arr_sorted)))
    for vctr in obs_arr_sorted:
        if fnmatch.fnmatch(str(vctr[0]), '*_T?????'):
        # if ('LC08' not in vctr[0]) & ('LE07' not in vctr[0]):
            nir = np.append(nir, float(vctr[4+i_s2_nir]) / 10000.)
            red = np.append(red, float(vctr[4+i_s2_red]) / 10000.)
            blue = np.append(blue, float(vctr[4+i_s2_blue]) / 10000.)
        elif 'LC08' in vctr[0]:
            nir = np.append(nir, float(vctr[4+i_l8_nir]))
            red = np.append(red, float(vctr[4+i_l8_red]))
            blue = np.append(blue, float(vctr[4+i_l8_blue]))
        elif 'LE07' in vctr[0]:
            nir = np.append(nir, float(vctr[4+i_l7_nir]))
            red = np.append(red, float(vctr[4+i_l7_red]))
            blue = np.append(blue, float(vctr[4+i_l7_blue]))
        elif 'LT05' in vctr[0]:
            nir = np.append(nir, float(vctr[4+i_l5_nir]))
            red = np.append(red, float(vctr[4+i_l5_red]))
            blue = np.append(blue, float(vctr[4+i_l5_blue]))
        else:
            raise Exception, 'check'

    ndvi = ((nir - red) / (nir + red))
    np.nan_to_num(ndvi, copy=False)

    zero_msk = np.where((blue == 0.0) & (red == 0.0))
    hot = (blue - 0.5 * red - 0.08)
    hot[zero_msk] = 0.00
    hot_ok = np.where((hot <= hot_crit) & (hot != 0.000))

    #todo make flexible to return single band vals too

    doys = datetime2doy(time_arr_sorted)

    profile = eePixelTimeSeries(ObsTimeSeries = ndvi,
                                ObsType = 'NDVI',
                                acq_id = acq_ids_sorted,
                                sensor = sensor_id_arr_sorted,
                                date_time = time_arr_sorted,
                                date_doy = doys,
                                startDate = startDate,
                                endDate = endDate,
                                ObsHOT = hot,
                                ObsMask = hot_ok,
                                ObsMaskType = 'hot ok',
                                utm_x = utm_x,
                                utm_y = utm_y,
                                lat=lat,
                                lon=lon,
                                pixel_size=pixel_size,
                                POI_ID = POI_ID)
    return profile


def plotEEtimeSeriesAndImageChips(plotProfile=True,
                                  plotImageChips=True,
                                  timeSeriesProfile=None,
                                  imageChips=None,
                                  modis_profile=None,
                                  filtered_profile=False,
                                  verbose=False,
                                  outdir=None,
                                  hot_crit=0.02,
                                  monthly_chips = 3): # HOT threshold, for SR use 0.00 for TOA use 0.02 or higher

    assert outdir is not None
    import matplotlib.dates as mdates

    plotProfile = None if plotProfile == False else plotProfile
    plotImageChips = None if plotImageChips == False else plotImageChips

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    yearsFmt = mdates.DateFormatter('\n%Y')  # add some space for the year label
    #dts = s.index.to_pydatetime()


    crosshair = True
    profileRows = 3
    markersize = 4

    # plt.style.use('seaborn-white')
    plt.rcParams['font.serif'] = 'Verdana'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlepad'] = 3
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['legend.columnspacing'] = 0.1
    plt.rcParams['legend.handletextpad'] = 0.1
    plt.rcParams['legend.markerscale'] = 0.8
    plt.rcParams['legend.borderpad'] = 0.2
    plt.rcParams['figure.titlesize'] = 12
    #plt.rcParams['figure.subplot.hspace'] = 0.9
    # plt.rcParams['figure.subplot.wspace'] = 0.1
    # plt.rcParams['figure.subplot.bottom'] = 0.99
    # plt.rcParams['figure.subplot.top'] = 0.90
    plt.rcParams['axes.linewidth'] = 0.15
    plt.rcParams['lines.markersize'] = markersize
    plt.rcParams['lines.linewidth'] = 0.65
    plt.rcParams['grid.color'] = '0.5'
    plt.rcParams['grid.linestyle'] = 'solid'
    plt.rcParams['grid.linewidth'] = 0.2

    if plotProfile:
        assert timeSeriesProfile is not None

    if plotImageChips:

        assert imageChips is not None

        if True:
            # exclude image chips whose center pixel is masked or unreliable
            hot_ok = np.where((timeSeriesProfile.ObsHOT <= hot_crit) & (timeSeriesProfile.ObsHOT != 0.0))
            imageChips = [c for c in imageChips if c.date_time in timeSeriesProfile.date_time[hot_ok]]

        nChips = len(imageChips)

        if ((nChips > 40) and (monthly_chips is not None)):
            print '{0} image chips retrieved - reducing to monthly best...'.format(nChips)
            selected = reduceImageChipTimeSeries(imageChips, monthly_chips=monthly_chips, verbose=verbose)
            nChips = len(selected)
            imageChips = selected


        # determine number of columns and rows for image chip grid
        maxColumns = 10 # 8
        div = nChips / maxColumns + 1

        if nChips % 2 == 0:
            nCols = int(nChips / float(div))
            if nCols * div < nChips:
                nCols += 1
            nRows = div
        else:
            nCols = int((nChips + 1) / float(div))
            if nCols * div < nChips:
                nCols += 1
            nRows = div

        if nRows > 6:
            profileRows += 1

    elif not plotImageChips:
        profileRows = 2
        nCols = 6



    if plotProfile is not None and plotImageChips is not None:
        #test: add one row for extract space for top x-axis labels
        nRows += 1

        # profile & imageChips
        fig = plt.figure(figsize=(nCols*1.5, nRows*1.5))  # 5
        # fig = plt.figure(figsize=(9.5, 3.5))#5
        hratios = []
        hratios.extend(np.repeat(1,profileRows))
        hratios.extend(np.repeat(.6, 1)) # row for extract space for top x-axis labels
        # hratios.extend(np.repeat(1, nRows - 1))
        hratios.extend(np.repeat(float(nCols+1)/float(nRows), nRows-1))#2
        print(float(nCols+1)/float(nRows))
        gs = gridspec.GridSpec(nRows + profileRows, nCols, height_ratios=hratios)
        gs.update(wspace=0.05, hspace=0.05)#(wspace=0.0, hspace=0.6)


    elif plotProfile is not None and plotImageChips is None:
        # only profile
        fig = plt.figure(figsize=(16.0, 4.0))
        gs = gridspec.GridSpec(profileRows, nCols)

    elif plotProfile is False and plotImageChips is not None:
        # only image chips
        fig = plt.figure(figsize=(16.0, 4.0))
        gs = gridspec.GridSpec(nRows, nCols)

    if plotProfile is not None:

        axp = plt.subplot(gs[:profileRows, :])

        if modis_profile is not None:
            axp.plot(modis_profile.date_time.astype(datetime), modis_profile.ObsTimeSeries, linestyle='', marker='o',
                     markerfacecolor='0.75', markeredgecolor='0.2', markersize=min(markersize-2,2.5), linewidth=0)

        # is2 = np.where(timeSeriesProfile.sensor == 'S2')
        # il8 = np.where(timeSeriesProfile.sensor == 'L8')
        # il7 = np.where(timeSeriesProfile.sensor == 'L7')
        # il5 = np.where(timeSeriesProfile.sensor == 'L5')

        hot_ok = np.where((timeSeriesProfile.ObsHOT <= hot_crit)&(timeSeriesProfile.ObsHOT != 0.0))

        if filtered_profile:
            savgol_profile = savitzky_golay(timeSeriesProfile.ObsTimeSeries[hot_ok], window_size=11, order=2, deriv=0)
            axp.plot(timeSeriesProfile.date_time[hot_ok].astype(datetime), savgol_profile, linestyle='-', marker='',color='0.85', linewidth=2, label='savgol')

        try:
            iChip = np.array([np.where(timeSeriesProfile.date_time == c.date_time)[0] for c in imageChips])
        except:
            iChip = np.array([np.where(timeSeriesProfile.date_time == c[0].date_time)[0] for c in imageChips])

        if not filtered_profile:
            axp.plot(timeSeriesProfile.date_time[hot_ok].astype(datetime), timeSeriesProfile.ObsTimeSeries[hot_ok], linestyle='-', marker='', color='0')

        sensrs = ['S2','L8','L7','L5']
        symbls = ['o','D','D','D']
        colrs = ['red','green','blue','yellow']

        for i, snsr in enumerate(sensrs):

            inds_good = np.where((timeSeriesProfile.sensor == snsr)&((timeSeriesProfile.ObsHOT <= hot_crit)&(timeSeriesProfile.ObsHOT != 0.0)))
            inds_baad = np.where((timeSeriesProfile.sensor == snsr)&((timeSeriesProfile.ObsHOT > hot_crit)|(timeSeriesProfile.ObsHOT == 0.0)))

            axp.plot(timeSeriesProfile.date_time[inds_good].astype(datetime), timeSeriesProfile.ObsTimeSeries[inds_good],
                     linestyle='', marker=symbls[i], markerfacecolor=colrs[i], markeredgecolor='0', linewidth=0.25, label=snsr if len(inds_good[0]) > 0 else None)

            axp.scatter(timeSeriesProfile.date_time[inds_baad].astype(datetime), timeSeriesProfile.ObsTimeSeries[inds_baad], marker=symbls[i], c=colrs[i],
                        edgecolors='0', linewidths=0.25, alpha=0.25)


        # axp.plot(timeSeriesProfile.date_time[is2].astype(datetime), timeSeriesProfile.ObsTimeSeries[is2], linestyle='', marker='o', markerfacecolor='red',
        #          markeredgecolor='0', linewidth=0.25, label='S2' if len(is2[0]) > 0 else None)
        #
        # axp.plot(timeSeriesProfile.date_time[il8].astype(datetime), timeSeriesProfile.ObsTimeSeries[il8], linestyle='', marker='D', markerfacecolor='green',
        #          markeredgecolor='0', linewidth=0.25,  label='L8' if len(il8[0]) > 0 else None)
        #
        # axp.plot(timeSeriesProfile.date_time[il7].astype(datetime), timeSeriesProfile.ObsTimeSeries[il7], linestyle='', marker='D', markerfacecolor='blue',
        #          markeredgecolor='0', linewidth=0.25,  label='L7' if len(il7[0]) > 0 else None)
        #
        # axp.plot(timeSeriesProfile.date_time[il5].astype(datetime), timeSeriesProfile.ObsTimeSeries[il5], linestyle='',
        #          marker='D', markerfacecolor='yellow',markeredgecolor='0', linewidth=0.25, label='L5' if len(il5[0]) > 0 else None)

        if plotImageChips:
            axp.plot(timeSeriesProfile.date_time[iChip].astype(datetime), timeSeriesProfile.ObsTimeSeries[iChip], linestyle='', marker='s',
                     markerfacecolor='None', markeredgecolor='0.65', markersize=markersize+2.5, linewidth=0.1)#, label='ImageChip')

        if modis_profile is not None:
            axp.plot([0], [0], linestyle='', marker='o', markerfacecolor='0.75', markeredgecolor='0.2', markersize=markersize-2.5, linewidth=0.001, label='MODIS')

        axp.set_xlim(np.datetime64(timeSeriesProfile.startDate).astype(datetime), np.datetime64(timeSeriesProfile.endDate).astype(datetime))

        axp.set_ylim(0, 1)
        axp.set_ylabel('{0}-({1}x{1})'.format(timeSeriesProfile.ObsType,timeSeriesProfile.mean_window) if timeSeriesProfile.ObsType == 'NDVI_MEAN' else timeSeriesProfile.ObsType)

        # format the ticks
        axp.xaxis.set_minor_locator(months)
        axp.xaxis.set_minor_formatter(monthsFmt)
        plt.setp(axp.xaxis.get_minorticklabels(), rotation=0)
        axp.xaxis.set_major_locator(years)
        axp.xaxis.set_major_formatter(yearsFmt)
        #plt.xticks(np.arange(from_date, to_date, dtype='datetime64[M]').astype(datetime))

        axp.grid(which='both', axis='both')
        # axp.xaxis.grid(True)
        # axp.yaxis.grid(True)

        axp.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=4)

        if plotImageChips is True:
            npx = imageChips[0].image_chip.shape[0]
            axp.set_title("{0}, {1}x{1} px @ {2}m, (X={3}, Y={4})".format(timeSeriesProfile.POI_ID,
                                                                          npx, imageChips[0].pixel_size,
                                                                          max(timeSeriesProfile.lon,timeSeriesProfile.utm_x),
                                                                          max(timeSeriesProfile.lat,timeSeriesProfile.utm_y)))
        else:
            profile_res = timeSeriesProfile.pixel_size if timeSeriesProfile.is_mean is False else (timeSeriesProfile.pixel_size * timeSeriesProfile.mean_window)
            axp.set_title("{0} @{1}m (X={2}, Y={3})".format(timeSeriesProfile.POI_ID, profile_res,
                                                      max(timeSeriesProfile.lon,timeSeriesProfile.utm_x),
                                                      max(timeSeriesProfile.lat,timeSeriesProfile.utm_y)))

        #plt.tight_layout() #todo PG commented out on June 8th 2018 due to tight layout error

    if plotImageChips is not None:
        ################################################################################################################
        ##
        ## assemble image chip time series figure
        ##
        ################################################################################################################

        npx = imageChips[0].image_chip.shape[0]

        count = 0
        for i in range(profileRows+1, (profileRows + nRows + 1)):
            for j in range(nCols):

                if count >= nChips:
                    break

                axp = plt.subplot(gs[i, j])

                axp.imshow(imageChips[count].image_chip, origin='upper', vmin=0, vmax=255, extent=(0, npx, npx, 0))#, aspect='auto')

                axp.set(adjustable='box-forced', aspect='equal')

                if crosshair:
                    axp.annotate('+', xy=(0.5, 0.5), color='w', va='center', ha='center', xycoords='axes fraction')

                axp.set_axis_off()

                t = plt.text(.5, .01, str(imageChips[count].date_time)[:10], horizontalalignment='center',
                             transform=axp.transAxes, fontsize=8) #.952

                t.set_path_effects([PathEffects.Stroke(linewidth=1, foreground='w'), PathEffects.Normal()])
                count += 1

    info_str = ''

    try:
        if timeSeriesProfile.POI_ID is not None:
            info_str += 'ID-{0}'.format(timeSeriesProfile.POI_ID)
    except:
        info_str += 'Point'

    if timeSeriesProfile.is_mean is True:
        info_str += '_mean{0}x{1}'.format(timeSeriesProfile.mean_window,timeSeriesProfile.mean_window)

    if plotImageChips:
        info_str += '_{0}px@{1}m'.format(npx,timeSeriesProfile.pixel_size)
    else:
        info_str += '@{0}m'.format(timeSeriesProfile.pixel_size)

    if plotImageChips and imageChips[0].rgb_combination is not None:
        i = 0 if len(imageChips[0].rgb_combination) == 3 else 1
        info_str += '_RGB-{0}-{1}-{2}'.format(imageChips[0].rgb_combination[i+0],imageChips[0].rgb_combination[i+1],imageChips[0].rgb_combination[i+2])

    info_str += '_{0}-{1}'.format(timeSeriesProfile.startDate,timeSeriesProfile.endDate)

    plot_outname = '{0}\ImageChipSeries_{1}.png'.format(outdir, info_str)

    plt.savefig(plot_outname, dpi=500, format='PNG', bbox_inches='tight')

    plt.close()

    return plot_outname

#########################################################################################################
###########      Above: Image chip and image time series extraction          ##################################
###########      Below: Compositing functionality                            #################################################################
#########################################################################################################

def get_xyz_maptile_URL(image, visParams):
    MapTile = image.getMapId(visParams)
    URL = 'https://earthengine.googleapis.com/map/{0}/%7Bz%7D/%7Bx%7D/%7By%7D?token={1}'.format(MapTile['mapid'] ,MapTile['token'])
    # URL = "https://earthengine.googleapis.com/map/{0}/${1}/${2}/${3}?token={4}".format(MapTile['mapid'], 'z', 'x', 'y',MapTile['token'])#GL
    # URL = "https://earthengine.googleapis.com/map/{0}/{1}/{2}/{3}?token={4}".format(MapTile['mapid'], 'z', 'x', 'y',MapTile['token'])
    #URL = ee.String("https://earthengine.googleapis.com/map/"+MapTile['mapid']+"/{z}/{x}/{y}?token="+MapTile['token'])
    return URL

def s2_cloudmasking(image):
    cld = image.select('B2').multiply(image.select('B9')).divide(1e4)
    p1 = image.select('B2').divide(1e4)
    p2 = image.select('B4').divide(1e4).multiply(0.5)
    HOT = p1.subtract(p2).subtract(ee.Image(0.08))
    nirGTswir = image.select('B8').gt(image.select('B11'))
    cloud = cld.gt(125).And(HOT.gt(0.0)).And(nirGTswir.eq(1))
    scale = ee.Image.pixelArea().sqrt()
    buffer = cloud.updateMask(cloud).fastDistanceTransform(500, 'pixels').sqrt().multiply(scale)
    clouds = cloud.Not().updateMask(buffer.gt(150))
    shadows = projectShadows(clouds, image.get('MEAN_SOLAR_AZIMUTH_ANGLE'), image.get('MEAN_INCIDENCE_ZENITH_ANGLE_B10'))
    locmay = image.mask().reduce('min').And(clouds).And(shadows)
    return image.updateMask(locmay)
    #mask = cloud.eq(1).And(shadow.eq(1)).And(snow.eq(1))

def projectShadows(cloudMask, sunAzimuth, offset):
    azimuth = ee.Number(sunAzimuth).multiply(math.pi/180.0).add(math.pi/2.0)
    x = azimuth.cos().multiply(15.0).round()
    y = azimuth.sin().multiply(15.0).round()
    shadow = cloudMask.translate(x.multiply(ee.Number(offset)), y.multiply(ee.Number(offset)), 'pixels')
    return shadow

#***********************************************************************************************************************
#***********************************************************************************************************************
#*********************************** ASAP Compositing functions ********************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************
#***********************************************************************************************************************

def add_DOY_band(img):
    fc = ee.Feature(img)
    ft_paint = fc.set('DOY', ee.Date(fc.get('system:time_start')).getRelative('day', 'year').add(1))
    doy_band = ee.Image().toUint16().paint(ft_paint, 'DOY').rename('DOY')
    doy_band = doy_band.updateMask(img.select('B2'))
    return img.addBands(doy_band)

def add_deltaDOY_band(image_col, target_doy):
    '''
    Calculates the day difference to targtet day and add that as one layer per image to the collection
    :return: image collection with delta doy band added
    '''
    def addDeltaDOY(img):
        day = ee.Image(ee.Date(img.get('system:time_start')).getRelative('day', 'year').add(1 - target_doy)).abs().int16()
        return img.addBands(day.rename(['DELTA_DOY']))
    return (image_col.map(addDeltaDOY))

def add_HOT_band(image, fScale=1e4):
    '''
    Calculate Haze OPtimized Transformation
    For S2 apply sclaing by deviding by 10000
    For Landsat use scaling of 1
    :param image:
    :param fScale:
    :return:
    '''
    p1 = image.select('B2').divide(fScale)
    p2 = image.select('B4').divide(fScale).multiply(0.5)
    HOT = p1.subtract(p2).subtract(ee.Image(0.08))
    HOT = HOT.updateMask(image.select('B2'))
    return image.addBands(HOT.rename("HOT"))

def add_CldDistance_band(image):
    targetMask = image.mask().Not().clip(image.geometry())
    scale = ee.Image.pixelArea().sqrt()
    srs_img = image.select('B2').projection()
    CldShdDistance = targetMask.fastDistanceTransform(1000, 'pixels').sqrt().multiply(scale).clip(image.geometry(None, srs_img, None))
    CldShdDistance = CldShdDistance.updateMask(image.select('B2'))
    return image.addBands(CldShdDistance.rename("DISTANCE"))

def add_DOY_score(img_col):
    extrema = ee.Image(img_col.reduce(ee.Reducer.minMax()))
    iMin = extrema.select('DELTA_DOY_min')
    iMax = extrema.select('DELTA_DOY_max')
    def map_DOY_score(img):
        delta_doy = ee.Image(img).select('DELTA_DOY')
        score = ee.Image(delta_doy.subtract(iMax)).abs().multiply((ee.Image(1).divide(iMax.subtract(iMin))))
        #var score = delta_doy.subtract(iMin).abs().divide(range);
        score = score.rename('SCORE_DOY')
        return img.addBands(score)
    return(img_col.map(map_DOY_score))

def add_DISTANCE_score(img_col):
    extrema = ee.Image(img_col.reduce(ee.Reducer.minMax()))
    iMax = extrema.select('DISTANCE_max')
    def map_distance_score(img):
        score = img.select('DISTANCE').divide(iMax)
        score = ee.Image(score).rename('SCORE_DISTANCE')
        return img.addBands(score)
    return(img_col.map(map_distance_score))

def add_HOT_score(img_col):
    extrema = ee.Image(img_col.reduce(ee.Reducer.minMax()))
    iMin = extrema.select('HOT_min')
    iMax = extrema.select('HOT_max')
    range = iMax.subtract(iMin)
    def map_hot_score(img):
        hot = ee.Image(img).select('HOT')
        score = hot.subtract(iMin).abs().divide(range)
        score = score.rename('SCORE_HOT')
        return img.addBands(score)
    return(img_col.map(map_hot_score))

def get_total_score(img):
    total_score = ee.Image(img).select(['SCORE_DOY', 'SCORE_DISTANCE', 'SCORE_HOT']).reduce('sum')
    return img.addBands(total_score.rename('TOTAL_SCORE'))

def get_weighted_total_score(imgCol, weights):
    def applyWeightsToScores(img):
        wDOY = img.select(['SCORE_DOY']).multiply(weights['DOY'])
        wDST = img.select(['SCORE_DISTANCE']).multiply(weights['CldShdDist'])
        wHOT = img.select(['SCORE_HOT']).multiply(weights['HOT'])
        TOTAL_SCORE = ee.Image.cat([wDOY,wDST,wHOT]).reduce('sum')
        return img.addBands(TOTAL_SCORE.rename('TOTAL_SCORE'))
    return imgCol.map(applyWeightsToScores)

def get_nObs_band(img_col):
    img_col_band = img_col.select('B2')
    nObs = img_col_band.reduce(ee.Reducer.count())
    nObs = nObs.rename('nObs')
    return nObs

def extract_point_values(xycoords, read_image, at_scale=20):
    assert isinstance(xycoords, list)
    assert len(xycoords) == 2
    at_point = ee.Geometry.Point(xycoords)
    vals = read_image.reduceRegion('min', at_point, at_scale, 'epsg:4326').getInfo()
    return vals

def get_MinMax_values(image, band, scale=None, maxPixel=10000000):
    image_band = image.select(band)
    if scale is None:
        scale = image_band.projection().nominalScale().getInfo()
    aoi = image_band.geometry()
    #reduceRegion(reducer, geometry, scale, crs, crsTransform, bestEffort, maxPixels, tileScale)
    min = ee.Number(image_band.reduceRegion('min', aoi, scale, None, None, True, maxPixel, 1.).get(band))
    max = ee.Number(image_band.reduceRegion('max', aoi, scale, None, None, True, maxPixel, 1.).get(band))
    return [min.getInfo(), max.getInfo()]


def get_ndvi_stackimage(img_col):
    ndvi_list = []
    def get_ndvi_band(img):
        ndvi = img.normalizedDifference(['B8','B4'])
        ndvi = ndvi.rename('NDVI')
        time_stamp = img.get('system:time_start')
        ndvi = ndvi.set('system:time_start', time_stamp)
        ndvi_list.append(ndvi)
        return None
    _ = img_col.map(get_ndvi_band)
    ndvi_stack = ee.Image.cat(ndvi_list)
    return ndvi_stack

def get_ndvi_collection(img_col):
    #ndvi_list = []
    def get_ndvi(img):
        ndvi = img.normalizedDifference(['B8','B4'])
        ndvi = ndvi.rename('NDVI')
        time_stamp = img.get('system:time_start')
        ndvi = ndvi.set('system:time_start', time_stamp)
        return ndvi
    ndvi_col = img_col.map(get_ndvi)
    return ndvi_col

def get_MGRS_tile_bounds(image_band):
    scale = image_band.projection().nominalScale().getInfo()
    aoi = image_band.geometry(scale).bounds()
    return aoi

def get_target_doy_ranges(start='2018-07-01', end='2018-08-28', interval=10):
    tdoy_list = []
    epoch_start = ee.Date.parse('Y-M-d', start, None)
    doy_start = epoch_start.getRelative('day', 'year').add(1).getInfo()
    doy_end = doy_start + interval
    year_start = int(start[:4])
    year_end = int(end[:4])
    time_series_end = ee.Date.parse('Y-M-d', end, None);
    doy_final = time_series_end.getRelative('day', 'year').add(1).getInfo()
    count = 0
    while (doy_end <= doy_final):
        tdoy = math.floor(doy_start + ((doy_end - doy_start)/2))
        # tdoy_list.insert(count, ee.List([year_start, doy_start, int(tdoy), doy_end, year_end]))
        #tdoy_list.append(ee.List([year_start, doy_start, int(tdoy), doy_end, year_end]))
        tdoy_list.append([year_start, doy_start, int(tdoy), doy_end, year_end])
        doy_start = doy_start + interval
        doy_end = doy_end + interval
        count += 1
    return tdoy_list #ee.List(tdoy_list)

def get_pbc_for_tdoy_ranges(ranges, aoi, subset_bands=True):
    def map_composite_doy_ranges(iRange):
        iRange = ee.List(iRange)
        # val1 = ee.Number(iRange.get(0))
        # val2 = ee.Number(iRange.get(1))
        # print(val1, val2)
        # return [val1, val2]
        init_date_str1 = '{0}-01-01'.format(ee.Number(iRange.get(0)).getInfo())
        init_date_str2 = '{0}-01-01'.format(ee.Number(iRange.get(4)).getInfo())
        # interval_start = ee.Date.parse('Y-M-d', init_date_str1, None)
        # interval_end = ee.Date.parse('Y-M-d', init_date_str2, None)
        # interval_start = interval_start.advance(ee.Number(iRange.get(1)), 'day')
        # interval_end = interval_end.advance(ee.Number(iRange.get(3)), 'day')
        # tdoy = ee.Number(iRange.get(1))
        # if subset_bands:
        #     # todo: remove below, currently required due to User memory limit exceeded.
        #     s2_col = ee.ImageCollection('COPERNICUS/S2').filterBounds(aoi).filterDate(interval_start, interval_end).select('B8','B11','B4','B2','B9')
        # else:
        #     s2_col = ee.ImageCollection('COPERNICUS/S2').filterBounds(aoi).filterDate(interval_start,interval_end)
        # s2_col = s2_col.map(s2cldmasking)
        # s2_col = getdeltaDOY(s2_col, tdoy)
        # s2_col = s2_col.map(map_addHOT)
        # s2_col = s2_col.map(map_addCldShdDistImg)
        # s2_col = score_deltaDOY(s2_col)
        # s2_col = score_distance(s2_col)
        # s2_col = score_hot(s2_col)
        # s2_col = s2_col.map(map_add_DOY_band)
        # s2_col = s2_col.map(map_get_total_score)
        # s2_col_pbc = s2_col.qualityMosaic('TOTAL_SCORE')
        # s2_col_pbc = s2_col_pbc.addBands([get_nObs(s2_col)])
        # s2_col_pbc = s2_col_pbc.set('system:time_start', interval_start.millis())
        # return s2_col_pbc
    return ranges.map(map_composite_doy_ranges)

def make_pbc_for_daterange(iRange, aoi, subset_bands=True):
    init_date_str1 = '{0}-01-01'.format(iRange[0])
    init_date_str2 = '{0}-01-01'.format(iRange[4])
    interval_start = ee.Date.parse('Y-M-d', init_date_str1, None)
    interval_end = ee.Date.parse('Y-M-d', init_date_str2, None)
    interval_start = interval_start.advance(iRange[1], 'day')
    interval_end = interval_end.advance(iRange[3], 'day')
    tdoy = iRange[1]
    if subset_bands:
        # todo: remove below, currently required due to User memory limit exceeded.
        s2_col = ee.ImageCollection('COPERNICUS/S2').filterBounds(aoi).filterDate(interval_start, interval_end).select('B8','B11','B4','B2','B9')
    else:
        s2_col = ee.ImageCollection('COPERNICUS/S2').filterBounds(aoi).filterDate(interval_start,interval_end)
    s2_col = s2_col.map(s2_cloudmasking)
    s2_col = add_deltaDOY_band(s2_col, tdoy)
    s2_col = s2_col.map(add_HOT_band)
    s2_col = s2_col.map(add_CldDistance_band)
    s2_col = add_DOY_score(s2_col)
    s2_col = add_DISTANCE_score(s2_col)
    s2_col = add_HOT_score(s2_col)
    s2_col = s2_col.map(add_DOY_band)
    s2_col = s2_col.map(get_total_score)
    s2_col_pbc = s2_col.qualityMosaic('TOTAL_SCORE')
    s2_col_pbc = s2_col_pbc.addBands([get_nObs_band(s2_col)])
    s2_col_pbc = s2_col_pbc.set('system:time_start', interval_start.millis())
    return s2_col_pbc

def ConvertImageCollection2Image(imgCol):
    emptyImage = ee.Image()
    def combine(image, result):
        return ee.Image(result).addBands(image)
    multibandImage = ee.Image(imgCol.iterate(combine, emptyImage))#.select(1)
    return multibandImage

def eeExportImage2Drive(img, visParams, export_name, scale=None, _region=None):
    #img_vis = img.visualize(bands=['B8', 'B11', 'B4'], min=[250, 250, 0], max=[3200, 3200, 1800])
    # img_vis = img.visualize(visParams)

    def wrapper_visualize(img, visParams, _bands=None, _min=None, _max=None):
        # Utility to allow for dictionary vis paramters to be passed into function
        keys = visParams.keys()
        if 'bands' in keys:
            _bands = visParams['bands'].split(', ')
        if 'min' in keys:
            _min = [int(v) for v in visParams['min'].split(', ')]
        if 'max' in keys:
            _max = [int(v) for v in visParams['max'].split(', ')]
        # if 'format' in keys:
        #     _format = [visParams['format']]
        return img.visualize(bands=_bands, min=_min, max=_max)

    img_vis = wrapper_visualize(img, visParams)

    if scale is None:
        scale = img_vis.select('vis-red').projection().nominalScale().getInfo()
    #aoi = img_vis.geometry(scale).bounds()

    if _region is None:
        _region = img.geometry().bounds().getInfo()['coordinates'][0]
    else:
        _region = _region.geometry().bounds().getInfo()['coordinates'][0]

    task = ee.batch.Export.image.toDrive(image=img_vis,
                                         description=export_name,
                                         scale=scale,
                                         region=_region,
                                         maxPixels=1000000000,
                                         folder='GEE_Exports')#region=aoi - use clip instead?
    task.start()
    count=0
    print('exporting to Drive...')
    while task.status()['state'] in ['RUNNING', 'READY']:
        time.sleep(2)
        print('{0} - {1} - ({2} seconds passed...)'.format(task.status()['state'], export_name, (count*2)))
        count += 1
        if (task.status()['state'] in ['FAILED', 'CANCELLED', 'CANCEL_REQUESTED']):
            print(task.status())
            break
        elif (task.status()['state'] == 'COMPLETED'):
            print('export complete')
            return


def eeGetDownloadURL(img, filepath, scale=None, download=True):
    if scale is None:
        bandNames = img.bandNames().getInfo()
        scale = img.select(bandNames[2]).projection().nominalScale().getInfo()
    iURL = img.getDownloadURL({scale: scale})
    if download is False:
        return iURL
    else:
        urllib.urlretrieve(iURL, filepath)

def pbc_weights_selector(interval):
    if interval > 21:
        # for broad temporal intervals
        return {'DOY': 0.5, 'CldShdDist': 0.3, 'HOT': 0.2}
    else:
        # for narrow temporal intervals
        return {'DOY': 0.2, 'CldShdDist': 0.4, 'HOT': 0.4}

def get_single_asap_pbc_layer(start_date, end_date, region, sensor='S2', weights=None, add_DOY=None, add_nOBS=None):
    '''
    Function combines all steps to derive a single pbc (pixel-based composite) coverage
    By default assumes Sentinel2 but also works for Landsat 8 --> set sensor = 'L8'
    :param start_date:
    :param end_date:
    :param region:
    :param sensor:
    :param weights:
    :param add_DOY:
    :param add_nOBS:
    :return:
    '''
    assert sensor in ['S2','L8'], 'unknown sensor'

    if not isinstance(start_date, ee.Date): start_date = ee.Date.parse('Y-M-d', start_date)
    if not isinstance(end_date, ee.Date): end_date = ee.Date.parse('Y-M-d', end_date)

    if sensor == 'S2':
        s2_col = ee.ImageCollection('COPERNICUS/S2').filterBounds(region).filterDate(start_date, end_date)
        s2_col = s2_col.map(s2_cloudmasking)
    elif sensor == 'L8':
        s2_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA').filterBounds(region).filterDate(start_date, end_date)
        s2_col = s2_col.map(L8_TOA_masking)

    half_deltad = end_date.difference(start_date, 'day').divide(2).floor().getInfo()
    tdoy = start_date.advance(half_deltad, 'day').getRelative('day', 'year').add(1).getInfo()

    if add_DOY:
        s2_col = s2_col.map(add_DOY_band)

    s2_col = add_deltaDOY_band(s2_col, tdoy)
    s2_col = s2_col.map(add_HOT_band)
    s2_col = s2_col.map(add_CldDistance_band)

    s2_col = add_DOY_score(s2_col)
    s2_col = add_DISTANCE_score(s2_col)
    s2_col = add_HOT_score(s2_col)

    if weights is not None:
        s2_col = get_weighted_total_score(s2_col, weights)
    else: s2_col = s2_col.map(get_total_score)

    s2_col_pbc = s2_col.qualityMosaic('TOTAL_SCORE')

    if add_nOBS:
        s2_col_pbc = s2_col_pbc.addBands([get_nObs_band(s2_col)])
    # s2_col_pbc = s2_col_pbc.set('system:time_start', ee.Date.parse('Y-M-d', start_date).millis())

    return s2_col_pbc.clip(region)


def locate_main():
    pass
if __name__ == '__main__':

    # todo: allow for savitzky_golay profile in CSV

    visParams = {'bands': 'B8, B11, B4', 'min': '250, 250, 0', 'max': '3200, 3200, 1800', 'format': 'png'}
    visParams_L8 = {'bands': 'B5, B6, B4', 'min': '0.25, 0.25, 0.0', 'max': '0.3200, 0.3200, 0.1800', 'format': 'png'}
    weights = {'DOY' : 0.2, 'CldShdDist' : 0.4, 'HOT' : 0.4}
    gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')


    if True:
        # Michele / Guido: Example for running single PBC coverage and exporting to drive and getting XYZ tiles URL for QGIS
        run_name = 'Bay'  # MV'
        gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')
        aoi_gaul = ee.Feature(gaul1.filterMetadata('adm1_name', 'equals', "Bay").first())
        aoi_bounds = aoi_gaul.geometry().bounds()
        aoi = aoi_gaul.geometry()

        pbc = get_single_asap_pbc_layer('2018-05-01', '2018-05-10',sensor='L8',
                                        region=aoi, add_DOY=True, add_nOBS=True)

        print(get_xyz_maptile_URL(pbc.clip(aoi_gaul), visParams_L8))

        export_name = 'PBCIMG{0}'.format(run_name)
        eeExportImage2Drive(pbc.clip(aoi_gaul), visParams, export_name, scale=20, _region=aoi_gaul)

        print(01)




    if False:
        img = ee.Image('COPERNICUS/S2/20180719T103019_20180719T103020_T32TMR')
        minmax_band = get_MinMax_values(img, 'B11', 20)
        print(minmax_band)
        print(1)




    if False:
        request = {'asap1_id': '262', 'cloudpcent': '10', 'end_period': '2018-07-31', 'deltad': '15'}
        aoi_gaul = ee.Feature(gaul1.filterMetadata('asap1_id', 'equals', "262").first())
        pbc = get_single_asap_pbc_layer('2018-07-20', '2018-07-31', aoi_gaul.geometry(), weights=None)
        print(pbc)


    if False:
        # Example of profile and image chip extraction
        latlon = [12.76022, 41.89794]
        poi_id = 'RM'
        win_size = 900
        startdate = '2016-01-01'
        enddate = '2017-09-20'
        rgb_comb = ['nir', 'swir1', 'red'] # ['red', 'green', 'blue']

        if True:
            plot = create_TimeSeries_Plot(x=latlon[0], y=latlon[1], POI_ID=poi_id, windowSize=win_size, maxCloud=70,
                                          start_date=startdate, end_date=enddate, overplot_modis=True,
                                          filtered_profile=True, profile_only=False, profile_mean=False, profile_window=9,
                                          monthly_chips=1, sensors=['S2', 'L8'], pixel_size=20, hot_crit=0.02, HOT_cutoff=60,
                                          rgb_combination=rgb_comb, csv_output=True, csv_only=False, outdir=r'C:\TEMP\19September2018')#['nir', 'swir1', 'red']

        if True:
            png = get_individual_imageChips(x=latlon[0], y=latlon[1], POI_ID=poi_id, windowSize=win_size, maxCloud=70,
                                            start_date=startdate, end_date=enddate, monthly_chips=None,
                                            sensors=['S2', 'L8'], pixel_size=20, hot_crit=0.02, HOT_cutoff=60,
                                            rgb_combination=rgb_comb, outdir=r'C:\TEMP\19September2018')

        os.startfile(plot)
        print(1)

    if False:
        img = ee.Image('COPERNICUS/S2/20180719T103019_20180719T103020_T32TMR')
        fp = r"C:\Users\griffpa\Documents\Projects\ASAP\asap_pbc\New folder\py_pf_url_download_scale200.zip"
        eeGetDownloadURL(img, fp, scale=200, download=True)


    if False:
        # Guido / Michele: Example of how to run e.g. 10day composite time series and extract NDVI profile
        run_name = 'Bay'#MV'
        gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')

        #mgrs_ref = ee.Image('COPERNICUS/S2/20180719T103019_20180719T103020_T32TMR')
        #aoi = get_MGRS_tile_bounds(mgrs_ref.select('B11'))
        # gaul1 = gaul1.filterMetadata('adm1_name', 'equals', 'Mecklenburg-Vorpommern')
        # aoi_gaul = gaul1.filterMetadata('adm1_name', 'equals', 'Mecklenburg-Vorpommern')
        # aoi_gaul = gaul1.filterMetadata('adm1_name', 'equals', 'Mecklenburg-Vorpommern')#.union()
        aoi_gaul = ee.Feature(gaul1.filterMetadata('adm1_name', 'equals', "Bay").first())
        # aoi_gaul = ee.Feature(gaul1.filterMetadata('adm1_name', 'equals', 'Mecklenburg-Vorpommern').first())
        aoi_bounds = aoi_gaul.geometry().bounds()
        aoi = aoi_gaul.geometry()

        ranges = get_target_doy_ranges(start='2018-01-01', end='2018-09-20', interval=10)

        cts = []
        for iRange in ranges:

            ipbc = make_pbc_for_daterange(iRange, aoi_bounds)
            cts.append(ipbc)

            if False: # Export via Drive
                print(get_xyz_maptile_URL(ipbc.clip(aoi_gaul), visParams))
                export_name = 'PBCIMG{0}{1}scale20'.format(str(iRange[2]).zfill(3),iRange[0])
                eeExportImage2Drive(ipbc.clip(aoi_gaul), visParams, export_name, scale=20, _region=aoi_gaul)
            if False: # Export via TMS
                MapTile = ipbc.getMapId(visParams)
                rpt = gtms.reprojectLonLat(13.105764, 47.213995, 32631)
                xres = 20
                dx = 100
                export_name = '{1}{0}Chip'.format(str(iRange[2]).zfill(3), iRange[0])
                print('exporting: {0}'.format(export_name))
                export_img = gtms.getWarpedGEETiles(MapTile['mapid'], MapTile['token'], rpt[0] - xres * dx / 2, rpt[1] + xres * dx / 2, xres, xres, dx, dx, 32631)
                gtms.datasourceToPNG(export_img, r'C:\TEMP\19September2018\{0}.png'.format(export_name), worldfile=False)

        pbcTS = ee.ImageCollection.fromImages(cts)

        col_ndvi = get_ndvi_collection(pbcTS)

        col_ndvi_img = ConvertImageCollection2Image(col_ndvi)

        vals = extract_point_values([43.18468, 2.701204], col_ndvi_img)

        print(vals)

        print('stop')


    if False:
        s2_1 = ee.Image('COPERNICUS/S2/20180704T103021_20180704T103023_T32TMQ')
        s2_2 = ee.Image('COPERNICUS/S2/20180706T102019_20180706T102724_T32TMQ')
        s2_3 = ee.Image('COPERNICUS/S2/20180709T103019_20180709T103418_T32TMQ')
        s2_4 = ee.Image('COPERNICUS/S2/20180714T103021_20180714T103023_T32TMQ')
        s2_5 = ee.Image('COPERNICUS/S2/20180719T103019_20180719T103020_T32TMQ')
        s2ImgCol = ee.ImageCollection.fromImages([s2_1, s2_2, s2_3, s2_4, s2_5])

        s2ImgCol = s2ImgCol.map(s2_cloudmasking)
        # print(s2ImgCol.first().bandNames().getInfo())

        s2ImgCol = s2ImgCol.map(add_DOY_band)

        s2ImgCol = add_deltaDOY_band(image_col=s2ImgCol, target_doy=193)

        s2ImgCol = s2ImgCol.map(add_HOT_band)

        s2ImgCol = s2ImgCol.map(add_CldDistance_band)

        s2ImgCol = add_DOY_score(s2ImgCol)

        s2ImgCol = add_DISTANCE_score(s2ImgCol)

        s2ImgCol = add_HOT_score(s2ImgCol)

        # s2ImgCol = s2ImgCol.map(map_get_total_score)
        s2ImgCol = get_weighted_total_score(s2ImgCol, weights)

        s2ImgCol_pbc = s2ImgCol.qualityMosaic('TOTAL_SCORE')

        s2ImgCol_pbc = s2ImgCol_pbc.addBands([get_nObs_band(s2ImgCol)])

        url_img = get_xyz_maptile_URL(s2ImgCol_pbc, visParams)
        url_doy = get_xyz_maptile_URL(s2ImgCol_pbc, {'bands': 'DOY', 'min': '185', 'max': '200', 'palette': "08ff1d, ff1306"})
        url_obs = get_xyz_maptile_URL(s2ImgCol_pbc, {'bands': 'nObs', 'min': '0', 'max': '6', 'palette': '000000, 2892C7, FAFA64, E81014'})

        MapTile = s2ImgCol_pbc.getMapId(visParams)

        rpt = gtms.reprojectLonLat(8.0585, 44.72273, 32631)
        xres = 20
        dx = 10000

        if True:
            # export as TMS using Guido's functions
            print('exporting img...')
            export_img = gtms.getWarpedGEETiles(MapTile['mapid'], MapTile['token'], rpt[0] - xres * dx / 2, rpt[1] + xres * dx / 2, xres, xres, dx, dx, 32631)
            gtms.datasourceToPNG(export_img, r'C:\TEMP\test_export_tms_img.png')

            token = url_doy.split('token=')[1]
            mapid = url_doy.split('/')[4]

            print('exporting doy...')
            export_doy = gtms.getWarpedGEETiles(url_doy.split('/')[4], url_doy.split('token=')[1], rpt[0] - xres * dx / 2, rpt[1] + xres * dx / 2, xres, xres, dx, dx, 32631)
            gtms.datasourceToPNG(export_doy, r'C:\TEMP\test_export_tms_doy.png')

            print('exporting obs...')
            export_obs = gtms.getWarpedGEETiles(url_obs.split('/')[4], url_obs.split('token=')[1], rpt[0] - xres * dx / 2, rpt[1] + xres * dx / 2, xres, xres, dx, dx, 32631)
            gtms.datasourceToPNG(export_obs, r'C:\TEMP\test_export_tms_obs.png')

            vals = extract_point_values([8.0585, 44.72273], s2ImgCol_pbc)



    if False:
        visParams = {'bands':'B8, B11, B4', 'min':'250, 250, 0', 'max':'3200, 3200, 1800', 'format': 'png'}
        test_img = ee.Image('COPERNICUS/S2/20180719T103019_20180719T103020_T32TMR')
        test_img_masked = s2_cloudmasking(test_img)
        url = get_xyz_maptile_URL(test_img_masked, 'test_url', visParams)
        # https://earthengine.googleapis.com/map/48a92c3a3bbb1e5e894a7e8eb065c019/%7Bz%7D/%7Bx%7D/%7By%7D?token=f8336c391a417c9753388973fdf136d2
        print(url)



    if False:

        POI_ID = "Ispra"
        xlatlon = 8.63137
        ylatlon = 45.801189

        png = get_individual_imageChips(x=xlatlon, y=ylatlon, POI_ID=POI_ID, windowSize=150, maxCloud=70,
                                        start_date='2017-01-01', end_date='2018-07-18', monthly_chips=None,
                                        sensors=['S2','L8'], pixel_size=10, hot_crit=0.1, HOT_cutoff=60,
                                        rgb_combination=['nir', 'red', 'green'], outdir=r'C:\TEMP\renderChips')

        if False:
            plot = create_TimeSeries_Plot(x=xlatlon, y=ylatlon, POI_ID=POI_ID, windowSize=150, maxCloud=70,
                                          start_date='2017-01-01', end_date='2018-07-18', overplot_modis=False,
                                          filtered_profile=True, profile_only=False, profile_mean=False, profile_window=9, monthly_chips=None,
                                          sensors=['S2','L8'], pixel_size=10, hot_crit=0.1, HOT_cutoff=60,
                                          rgb_combination=['nir', 'red', 'green'], csv_output=True, outdir=r'C:\TEMP')

        print(plot)


    if False:
        fptxt = r"C:\Processing\test\ndvi_stack_T33PWP.txt"
        poi_content = np.genfromtxt(fptxt, dtype=np.datetime64, skip_header=0)
        doys = datetime2doy(poi_content)
        print(1)


    if False:
        ### test polygone AOI based profile extrqaction
        shpfile = r"C:\Processing\asap-hr-evolution\test_poly_CAM_Z32N.shp"

        polyAOI = shapefile_2_ftr_dict_func(shpfile)
        jsonAOI = convert_shp2geojson(shpfile)
        maskAOI = ee.FeatureCollection(polyAOI).reduceToImage(['var1'], 'max')

        s2img = ee.Image('COPERNICUS/S2/20180127T093241_20180127T094128_T32NPJ')

        s2img_clip = s2img.clip(polyAOI)

        url = s2img_clip.select('B8').getDownloadURL()

        test = s2img_clip.select('B8').reduce('mean')
        test2 = s2img.clip(polyAOI).select('B8').reduce('mean')

        s2imgColl= ee.ImageCollection.fromImages(s2img)

        url = s2imgColl.clipToCollection(polyAOI).getDownloadURL()

        test = s2img.clip(polyAOI.geometry())

        col2img = ee.Image(s2imgColl.filterBounds(polyAOI.geometry()))




        test = s2img.select('B11').clip(polyAOI).getInfo()

        url3 = s2img.select('B8', 'B11', 'B4').updateMask(maskAOI.gt(1)).clip(polyAOI).getDownloadURL()

        url = s2img.select('B08','B11','B04').clip(polyAOI).getDownloadURL()

        url2 = s2img.select('B08', 'B11', 'B04').reduce(polyAOI).getDownloadURL()

        test2 = s2img.clip(polyAOI).getInfo()

        # test = convert_shp2geojson(shpfile)
        #
        # jsonfile = r"C:\Users\griffpa\Documents\Projects\Cameroon\VHR-Acquisition\AOI\AOI.json"
        #
        # json_file = open(jsonfile, "r")
        # json_data = json.load(json_file)
        # json_file.close()
        # aoi = ee.Geometry.Polygon(json_data)
        # ee.Geometry(json_data)

    if False:
        POI_ID = 'grassland-DE-01'
        utmxy = [685026, 5432327]
        lonlat = convert_utm_to_latlon(utmxy[0], utmxy[1], src_epsg=32632)
        ListOfPOIs = list()
        POI = PointOfInterest(name="DE grassland TEST", LATLON_X=lonlat[0], LATLON_Y=lonlat[1])
        ListOfPOIs.append(POI)
        outdir = r"C:\TEMP"

    if False:
        outdir = r"C:\TEMP\Herve3"
        fp_poi_csv = r"C:\Users\griffpa\Documents\Projects\ASAP\Herve_POIs.csv"
        poi_content = np.genfromtxt(fp_poi_csv, dtype=np.object, delimiter=',', skip_header=1)
        ListOfPOIs = list()
        for i in range(len(poi_content)):
            POI = PointOfInterest(name=poi_content[i][0],LATLON_X=float(poi_content[i][2]), LATLON_Y=float(poi_content[i][1]))
            ListOfPOIs.append(POI)

    if False:
        outdir = r"C:\TEMP\Cameroon_Plots4"
        fp_poi_csv = r'C:\Users\griffpa\Documents\Projects\Cameroon\pilot_POIs_v6.csv'#plantation_POIs.csv'
        #fp_poi_csv = r'C:\Users\Patrick Griffiths\Google Drive\Cameroon\pilot_POIs_v3.csv'
        poi_content = np.genfromtxt(fp_poi_csv, dtype=np.object, delimiter=',', skip_header=1)
        ListOfPOIs = list()
        for i in range(len(poi_content)):
            POI = PointOfInterest(name=poi_content[i][0], LATLON_X=float(poi_content[i][2]),
                                  LATLON_Y=float(poi_content[i][1]))
            ListOfPOIs.append(POI)

    if False:
        # test : ensure that North is Up
        outdir = r"C:\TEMP"
        ListOfPOIs = list()
        POI = PointOfInterest(name="TEST CAM", LATLON_X=10.109058,LATLON_Y=3.572247)
        ListOfPOIs.append(POI)

    if False:
        fp_poi_csv = r'C:\Users\griffpa\Documents\HUB-GEO\Grassland-paper\samples_maehweiden_cent_n100.csv'
        outdir = r"C:\Users\griffpa\Documents\HUB-GEO\Grassland-paper\samples_maehweiden_cent_n100"
        poi_content = np.genfromtxt(fp_poi_csv, dtype=np.object, delimiter=',', skip_header=1)
        ListOfPOIs = list()
        target_epsg = 32632
        for i in range(len(poi_content)):

            #utmxy = convert_latlon_to_utm(lat=float(poi_content[i][3]), lon=float(poi_content[i][4]),trg_epsg=target_epsg)

            POI = PointOfInterest(name=poi_content[i][0],
                                  #LATLON_X=float(poi_content[i][4]),
                                  #LATLON_Y=float(poi_content[i][3]),
                                  UTM_X=float(poi_content[i][1]),UTM_Y=float(poi_content[i][2]))

            ListOfPOIs.append(POI)

        rectangle_sizes = [500,1000,1800,2100]#[900, 1200, 1600]

        from_date = '2016-01-01'
        to_date = '2017-01-01'

        # from_date = '2015-01-01'
        # to_date = '2018-03-01'
        maxCloud = 80
        isCenter = True
        t_epsg = 4326

        profile_mean = False
        profile_window = 9

        overplot_modis = True

        monthly_ranges = get_monthly_ranges(from_date=from_date, to_date=to_date)

        for POI in ListOfPOIs:

            POI_ID = POI.name
            latlon = [POI.LATLON_Y, POI.LATLON_X]
            umm_xy = [POI.UTM_X, POI.UTM_Y]

            for rectangle in rectangle_sizes:

                start_time = time.time()

                image_chips = []

                collections = ['COPERNICUS/S2', 'LANDSAT/LC08/C01/T1_TOA']
                band_lists = [['B8', 'B11', 'B4', 'B2'],['B5', 'B6', 'B4','B2']]

                for c, col in enumerate(collections):

                    if rectangle >= 1000:
                        # break down into monthly retrievals
                        for mrange in monthly_ranges:

                            m_start_data = mrange[0]
                            m_end_data = mrange[1]

                            ChipData = eeRetrieval(collection=col,
                                            maxCloud=maxCloud,
                                            startDate=m_start_data,
                                            endDate=m_end_data,
                                            lat=None,#latlon[0],
                                            lon=None,#latlon[1],
                                            utm_x=umm_xy[0],
                                            utm_y=umm_xy[1],
                                            epsg=target_epsg,
                                            rectangle=True,
                                            rectDist=rectangle,
                                            pixel_size=20,
                                            bandList=band_lists[c],
                                            bandNames=['nir', 'swir1', 'red', 'blue'],
                                            )

                            ChipData.define_geometry()
                            ChipData.get_data()

                            if ((ChipData.retrieval is None) or (len(ChipData.retrieval) == 1)):
                                continue

                            chips = ExtractImageChipsFromEERetrieval(ChipData,nb=len(band_lists[0]),use_HOT_Mask=True,HOT_cutoff=50)

                            if len(chips) == 0:
                                # empty retrieval could result again due to HOT masking, todo deal with this before
                                continue


                            image_chips.extend(chips)

                            ChipData = None

                    else:
                        # full time period retrieval
                        ChipData = eeRetrieval(collection=col,
                                               maxCloud=maxCloud,
                                               startDate=from_date,
                                               endDate=to_date,
                                               lat=None,  # latlon[0],
                                               lon=None,  # latlon[1],
                                               utm_x=umm_xy[0],
                                               utm_y=umm_xy[1],
                                               epsg=target_epsg,
                                               rectangle=True,
                                               rectDist=rectangle,
                                               pixel_size=20,
                                               bandList=band_lists[c],
                                               bandNames=['nir', 'swir1', 'red', 'blue'],
                                               )

                        ChipData.define_geometry()
                        ChipData.get_data()
                        chips = ExtractImageChipsFromEERetrieval(ChipData, nb=len(band_lists[0]), use_HOT_Mask=True,HOT_cutoff=50)
                        image_chips.extend(chips)
                        ChipData = None

                image_chips_sorted = sort_imageChips(image_chips)

                profile_collections = ['COPERNICUS/S2','LANDSAT/LC08/C01/T1_TOA','LANDSAT/LE07/C01/T1_TOA']
                profile_bandlists = [['B8', 'B4', 'B2'],['B5', 'B4', 'B2'],['B4', 'B3', 'B1']]
                profile_results = []

                for c, col in enumerate(profile_collections):

                    if profile_mean is True:
                        profile = eeRetrieval(collection=col,
                                              startDate=from_date, endDate=to_date,
                                              lat=None,  # latlon[0],
                                              lon=None,  # latlon[1],
                                              utm_x=umm_xy[0],
                                              utm_y=umm_xy[1],
                                              rectangle=True,
                                              rectDist=(profile_window*20),
                                              pixel_size=20,
                                              bandList=profile_bandlists[c],
                                              bandNames=['nir', 'red', 'blue'],
                                              epsg=target_epsg)
                                              # epsg=t_epsg)

                    else:
                        # extract pixel profile
                        profile = eeRetrieval(collection=col,
                                              startDate=from_date,endDate=to_date,
                                              lat=None,  # latlon[0],
                                              lon=None,  # latlon[1],
                                              utm_x=umm_xy[0],
                                              utm_y=umm_xy[1],
                                              pixel_size=20,
                                              bandList=profile_bandlists[c],
                                              bandNames=['nir', 'red', 'blue'],
                                              epsg=target_epsg)
                                              #epsg=t_epsg)

                    profile.define_geometry()
                    profile.get_data()

                    profile_results.append(profile)

                if profile_mean is True:
                    pixel_timeseries = ExtractAverageTimeSeriesProfileFromEEretrieval(profile_results)
                else:
                    pixel_timeseries = TimeSeriesProfileFromEERetrieval(S2=profile_results[0],
                                                                    L8=profile_results[1],
                                                                    L7=profile_results[2],
                                                                    POI_ID=POI_ID)

                if overplot_modis:

                    modis_colls = ['MODIS/006/MOD09Q1', 'MODIS/006/MYD09Q1']
                    modis_doy_colls = ['MODIS/006/MOD09A1','MODIS/006/MYD09A1']
                    modis_results = []
                    modis_DOYs = []

                    for iCol, col in enumerate(modis_colls):

                        modis = eeRetrieval(collection=col,
                                            startDate=from_date, endDate=to_date,
                                            lat=None,  # latlon[0],
                                            lon=None,  # latlon[1],
                                            utm_x=umm_xy[0],
                                            utm_y=umm_xy[1],
                                            pixel_size=250, bandList=['sur_refl_b01', 'sur_refl_b02'],
                                            bandNames=['red', 'nir'], epsg=target_epsg)#epsg=t_epsg)

                        DOYs = eeRetrieval(collection=modis_doy_colls[iCol],startDate=from_date, endDate=to_date,
                                           lat=None,  # latlon[0],
                                           lon=None,  # latlon[1],
                                           utm_x=umm_xy[0],
                                           utm_y=umm_xy[1],
                                           pixel_size=250,
                                           bandList=['DayOfYear'], bandNames=['DOY'], epsg=target_epsg)#epsg=t_epsg)

                        modis.define_geometry()
                        modis.get_data()

                        DOYs.define_geometry()
                        DOYs.get_data()

                        modis_results.append(modis)
                        modis_DOYs.append(DOYs)

                    modis_timeseries = ModisTimeSeriesProfileFromEERetrieval(modis_terra=modis_results[0],terra_doys=modis_DOYs[0],
                                                                             modis_aqua=modis_results[1],aqua_doys=modis_DOYs[1],
                                                                             POI_ID=POI.name)

                fig = plotEEtimeSeriesAndImageChips(plotProfile=True, plotImageChips=True,
                                                    timeSeriesProfile=pixel_timeseries,
                                                    imageChips=image_chips_sorted,
                                                    modis_profile = modis_timeseries,
                                                    outdir=outdir)

                print "Processing time: %s seconds ---" % (time.time() - start_time)

                if False:
                    ####################################################################################################################
                    #                                              average profile extraction                                          #
                    #                                                                                                                  #
                    ####################################################################################################################

                    from_date = '2016-01-01'
                    to_date = '2016-12-31'
                    maxCloud = 50
                    isCenter = True
                    t_epsg = 4326

                    profile_mean = True
                    profile_window = 9

                    profile_collections = ['COPERNICUS/S2', 'LANDSAT/LC08/C01/T1_TOA', 'LANDSAT/LE07/C01/T1_TOA']
                    profile_bandlists = [['B8', 'B4', 'B2'], ['B5', 'B4', 'B2'], ['B4', 'B3', 'B1']]
                    MeanProfile_results = []
                    PixelProfile_results = []

                    for c, col in enumerate(profile_collections):
                        MeanProfile = eeRetrieval(collection=col,
                                                  startDate=from_date, endDate=to_date,
                                                  utm_x=None, utm_y=None,
                                                  lat=lonlat[1], lon=lonlat[0],
                                                  rectangle=profile_mean,
                                                  rectDist=profile_window * 20,
                                                  pixel_size=20,
                                                  bandList=profile_bandlists[c],
                                                  bandNames=['nir', 'red', 'blue'],
                                                  epsg=t_epsg)

                        MeanProfile.define_geometry()
                        MeanProfile.get_data()

                        MeanProfile_results.append(MeanProfile)

                        PixelProfile = eeRetrieval(collection=col,
                                                   startDate=from_date, endDate=to_date,
                                                   lat=lonlat[1], lon=lonlat[0],
                                                   pixel_size=20,
                                                   bandList=profile_bandlists[c],
                                                   bandNames=['nir', 'red', 'blue'],
                                                   epsg=t_epsg)

                        PixelProfile.define_geometry()
                        PixelProfile.get_data()

                        PixelProfile_results.append(PixelProfile)

                    tsMean = ExtractAverageTimeSeriesProfileFromEEretrieval(MeanProfile_results)
                    tsPixel = TimeSeriesProfileFromEERetrieval(S2=PixelProfile_results[0], L8=PixelProfile_results[1],
                                                               L7=PixelProfile_results[2])

                    tsMean.ObsTimeSeries[tsMean.ObsMask]
                    tsPixel.ObsTimeSeries[tsPixel.ObsMask]

                ################################################################################################################

            if False:
                for rectangle in rectangle_sizes:

                    start_time = time.time()

                    # GET IMAGE CHIPS
                    S2Chips = eeRetrieval(collection = 'COPERNICUS/S2',
                                         maxCloud = maxCloud,
                                         startDate = from_date,
                                         endDate = to_date,
                                         utm_x = None,#utmxy[0],
                                         utm_y = None,#utmxy[1],
                                         lat=latlon[0],
                                         lon=latlon[1],
                                         rectangle=True,
                                         rectDist=rectangle,
                                         centerPOI = isCenter,
                                         pixel_size = 20,
                                         bandList = ['B8', 'B11', 'B4', 'B2'],
                                         bandNames = ['nir','swir1','red','blue'],
                                         epsg = t_epsg,
                                        )

                    S2Chips.define_geometry()
                    S2Chips.get_data()

                    print '{0} coordinates used to define rectangle'.format(len(S2Chips.eeGeometry.getInfo()['coordinates'][0]))

                    S2Chips.eeGeometry.getInfo()

                    S2_image_chips = ExtractImageChipsFromEERetrieval(S2Chips,
                                                                   nb = len(S2Chips.bandList),
                                                                   use_HOT_Mask=True,
                                                                   HOT_cutoff=50)

                    L8Chips = eeRetrieval(collection='LANDSAT/LC08/C01/T1_TOA',
                                          maxCloud=maxCloud,
                                          startDate=from_date,
                                          endDate=to_date,
                                          utm_x=None,  # utmxy[0],
                                          utm_y=None,  # utmxy[1],
                                          lat=latlon[0],
                                          lon=latlon[1],
                                          rectangle=True,
                                          rectDist=rectangle,
                                          centerPOI=isCenter,
                                          pixel_size=20,
                                          bandList=['B4', 'B3', 'B1','B6'],
                                          bandNames=['nir', 'red', 'blue','swir1'],
                                          epsg=t_epsg,
                                          )

                    L8Chips.define_geometry()
                    L8Chips.get_data()

                    L8_image_chips = ExtractImageChipsFromEERetrieval(L8Chips,
                                                                   nb=len(L8Chips.bandList),
                                                                   use_HOT_Mask=True,
                                                                   HOT_cutoff=60)

                    image_chips = []
                    image_chips.extend(S2_image_chips)
                    image_chips.extend(L8_image_chips)

                    image_chips_sorted = sort_imageChips(image_chips)



                    #  GET TIME SERIES PROFILE
                    S2profile = eeRetrieval(collection = 'COPERNICUS/S2',
                                            startDate =from_date,
                                            endDate=to_date,
                                            utm_x=None,  # utmxy[0],
                                            utm_y=None,  # utmxy[1],
                                            lat=latlon[0],
                                            lon=latlon[1],
                                            pixel_size=20,
                                            bandList=['B8', 'B4', 'B2'],
                                            bandNames=['nir', 'red', 'blue'],
                                            epsg=t_epsg,
                                            )

                    S2profile.define_geometry()
                    S2profile.get_data()

                    L8profile = eeRetrieval(collection='LANDSAT/LC08/C01/T1_TOA',
                                            startDate=from_date,
                                            endDate=to_date,
                                            utm_x=None,  # utmxy[0],
                                            utm_y=None,  # utmxy[1],
                                            lat=latlon[0],
                                            lon=latlon[1],
                                            pixel_size=30,
                                            bandList=['B5', 'B4', 'B2'],
                                            bandNames=['nir','red','blue'],
                                            epsg=t_epsg,
                                            )

                    L8profile.define_geometry()
                    L8profile.get_data()

                    L7profile = eeRetrieval(collection='LANDSAT/LE07/C01/T1_TOA',
                                            startDate=from_date,
                                            endDate=to_date,
                                            utm_x=None,  # utmxy[0],
                                            utm_y=None,  # utmxy[1],
                                            lat=latlon[0],
                                            lon=latlon[1],
                                            pixel_size=30,
                                            bandList=['B4', 'B3', 'B1'],
                                            bandNames=['nir', 'red', 'blue'],
                                            epsg=t_epsg,
                                            )

                    L7profile.define_geometry()
                    L7profile.get_data()

                    pixel_timeseries = TimeSeriesProfileFromEERetrieval(S2=S2profile,
                                                                        L8=L8profile,
                                                                        L7=L7profile,
                                                                        POI_ID = POI_ID)

                    fig = plotEEtimeSeriesAndImageChips(plotProfile=True, plotImageChips=True,
                                                        timeSeriesProfile=pixel_timeseries,
                                                        imageChips=image_chips_sorted,
                                                        outdir = outdir)

                    print "Processing time: %s seconds ---" % (time.time() - start_time)

