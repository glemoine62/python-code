import math
import ee

def get_nObs_band(img_col):
    img_col_band = img_col.select('B2')
    nObs = img_col_band.reduce(ee.Reducer.count())
    nObs = nObs.rename('nObs')
    return nObs

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
    targetMask = image.mask().Not() # .clip(image.geometry())
    scale = ee.Image.pixelArea().sqrt()
    srs_img = image.select('B2').projection()
    CldShdDistance = targetMask.fastDistanceTransform(1000, 'pixels').sqrt().multiply(scale) #.clip(image.geometry(None, srs_img, None))
    CldShdDistance = CldShdDistance.updateMask(image.select('B2'))
    return image.addBands(CldShdDistance.rename("DISTANCE"))

def add_DOY_score(img_col, extrema):
    #extrema = ee.Image(img_col.select('DELTA_DOY').reduce(ee.Reducer.minMax()))
    iMin = extrema.select('DELTA_DOY_min')
    iMax = extrema.select('DELTA_DOY_max')
    range = iMax.subtract(iMin)
    def map_DOY_score(img):
        delta_doy = ee.Image(img).select('DELTA_DOY')
        score = ee.Image(delta_doy.subtract(iMax)).abs().divide(range)
        #var score = delta_doy.subtract(iMin).abs().divide(range);
        score = score.rename('SCORE_DOY')
        return img.addBands(score)
    return(img_col.map(map_DOY_score))

def add_DISTANCE_score(img_col, extrema):
    #extrema = ee.Image(img_col.select('DISTANCE').reduce(ee.Reducer.minMax()))
    iMax = extrema.select('DISTANCE_max')
    def map_distance_score(img):
        score = img.select('DISTANCE').divide(iMax)
        score = ee.Image(score).rename('SCORE_DISTANCE')
        return img.addBands(score)
    return(img_col.map(map_distance_score))

def add_HOT_score(img_col, extrema):
    #extrema = ee.Image(img_col.select('HOT').reduce(ee.Reducer.minMax()))
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


def pbc_weights_selector(interval):
    if interval > 21:
        # for broad temporal intervals
        return {'DOY': 0.5, 'CldShdDist': 0.3, 'HOT': 0.2}
    else:
        # for narrow temporal intervals
        return {'DOY': 0.2, 'CldShdDist': 0.4, 'HOT': 0.4}

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
    
    extrema = ee.Image(s2_col.select(['HOT', 'DISTANCE', 'DELTA_DOY']).reduce(ee.Reducer.minMax()))
    
    s2_col = add_DOY_score(s2_col, extrema)
    s2_col = add_DISTANCE_score(s2_col, extrema)
    s2_col = add_HOT_score(s2_col, extrema)

    if weights is not None:
        s2_col = get_weighted_total_score(s2_col, weights)
    else: s2_col = s2_col.map(get_total_score)

    s2_col_pbc = s2_col.qualityMosaic('TOTAL_SCORE')

    if add_nOBS:
        s2_col_pbc = s2_col_pbc.addBands([get_nObs_band(s2_col)])
    # s2_col_pbc = s2_col_pbc.set('system:time_start', ee.Date.parse('Y-M-d', start_date).millis())

    return s2_col_pbc.clip(region)