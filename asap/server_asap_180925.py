#!/usr/bin/env python
"""ASAP prototype for high resolution view of alert area."""

import json
import os
import random
import datetime

import config
import ee
import jinja2
import webapp2

from google.appengine.api import urlfetch

import asap_highres

jinja_environment = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))


class AsapInput(webapp2.RequestHandler):
    """Handler for the request parameters coming from ASAP."""

    def get(self):
        """Writes the index page to the response based on the template."""
        ee.Initialize(config.EE_CREDENTIALS)
        gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')
        # gaul1 = ee.FeatureCollection('ft:1LgaoATYMGfIAEA_3-3LcaqCA1LjYH1ru8jGtXoE3')
        asap1_id = int(self.request.get('asap1_id'))
        cloud_pcent = 10
        # Set default end date as 20 of the previous month
        today = datetime.date.today()
        deltad = 32
        if today.month == 1:
            end_period = today.replace(year=today.year - 1, month=12, day=20)
        else:
            end_period = today.replace(month=today.month - 1, day=20)

        start_period = end_period - datetime.timedelta(days=deltad)

        roi = ee.Feature(gaul1.filterMetadata('asap1_id', 'equals', asap1_id).first())
        asap0_name = roi.get('adm0_name').getInfo()
        asap1_name = roi.get('adm1_name').getInfo()
        center = roi.geometry().centroid().coordinates().getInfo()
        lon = (center[0])
        lat = (center[1])

        template = jinja_environment.get_template('templates/index.html')
        self.response.out.write(template.render({
            'asap1_id': asap1_id,
            'asap0_name': asap0_name,
            'asap1_name': asap1_name,
            'lon': center[0],
            'lat': center[1],
            'cloudpcent': cloud_pcent,
            'start_period': datetime.date.strftime(start_period, '%Y-%m-%d'),
            'end_period': datetime.date.strftime(end_period, '%Y-%m-%d'),
            'deltad': deltad,
        }))


class MainPage(webapp2.RequestHandler):
    """Handler for the main index page."""

    def get(self):
        """Writes the index page to the response based on the template."""
        template = jinja_environment.get_template('templates/index.html')
        self.response.out.write(template.render({
            'cache_bust': random.randint(0, 150000),
        }))


class GetMapData(webapp2.RequestHandler):
    """Handler for the layer selection."""

    def get(self):

        do_add_DOY_band = True  # ***PG Edit***
        do_add_nOBS_band = True  # ***PG Edit***

        # if use_asap_pbc is True, then use user defined compositing procedures
        # if use_asap_pbc is False, then the mosaic() reducer is used instead
        use_asap_pbc = self.request.get('use_asap_pbc')

        """Sets up the request to Earth Engine and returns the map information."""
        ee.Initialize(config.EE_CREDENTIALS)#todo PG reactivate
        ee.data.setDeadline(120000)

        try: # todo PG edit
            urlfetch.set_default_fetch_deadline(120000)
        except:
            pass

        gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')
        asap1_id = int(self.request.get('asap1_id'))
        cloud_pcent = int(self.request.get('cloudpcent'))

        deltad = int(self.request.get('deltad'))
        end_date = ee.Date(self.request.get('end_period'))
        start_date = end_date.advance(-deltad, 'day')

        roi = ee.Feature(gaul1.filterMetadata('asap1_id', 'equals', asap1_id).first())

        # start_date = ee.Date('2017-06-20') #ee.Date(datetime.datetime.now()).advance(-40, 'day')
        # end_date = ee.Date('2017-07-20') #ee.Date(datetime.datetime.now()).advance(-10, 'day')
        start_date_1Y = start_date.advance(-1, 'year')
        end_date_1Y = end_date.advance(-1, 'year')

        layers = []

        # Select S2
        s2_visualization_options = {
            'bands': ','.join(['B8', 'B11', 'B4']),
            'min': 0,
            'max': ','.join(['6000', '6000', '4000']),
            'format': 'png'
        }

        if use_asap_pbc is True:

            weights = asap_highres.pbc_weights_selector(interval=deltad)

            s2now = asap_highres.get_single_asap_pbc_layer(start_date,
                                                           end_date,
                                                           region=roi.geometry(),
                                                           weights=weights,
                                                           add_DOY=do_add_DOY_band,
                                                           add_nOBS=do_add_nOBS_band)# ***PG Edit***
        elif use_asap_pbc is False:

            s2nowCol = ee.ImageCollection('COPERNICUS/S2')\
                        .filterDate(start_date, end_date)\
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent)\
                        .map(asap_highres.s2_cloudmasking)\
                        .map(asap_highres.add_DOY_band)\
                        .filterBounds(roi.geometry())

            s2now = ee.Image(s2nowCol.mosaic()).clip(roi.geometry())

            s2now = s2now.addBands([asap_highres.get_nObs_band(s2nowCol).clip(roi.geometry())])  # ***PG Edit***


        s2now_visualization = s2now.getMapId(s2_visualization_options)

        s2now_nb = int(s2now.bandNames().size().getInfo())
        # Add the recent Sentinel-2 composite.
        if s2now_nb > 0:
            layers.append({
                'mapid': s2now_visualization['mapid'],
                'label': "Sentinel-2 ({} to {})".format(ee.Date.format(start_date, 'd MMM Y').getInfo(),
                                                        ee.Date.format(end_date, 'd MMM Y').getInfo()),
                'token': s2now_visualization['token']
            })

        if use_asap_pbc is True:

            s2lastyear = asap_highres.get_single_asap_pbc_layer(start_date_1Y,
                                                                end_date_1Y,
                                                                region=roi.geometry(),
                                                                weights=weights,
                                                                add_DOY=do_add_DOY_band,
                                                                add_nOBS=do_add_nOBS_band)  # ***PG Edit***

        elif use_asap_pbc is False:

            s2lastCol = ee.ImageCollection('COPERNICUS/S2') \
                          .filterDate(start_date_1Y, end_date_1Y) \
                          .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent) \
                          .map(asap_highres.s2_cloudmasking) \
                          .map(asap_highres.add_DOY_band) \
                          .filterBounds(roi.geometry())

            s2lastyear = ee.Image(s2lastCol.mosaic()).clip(roi.geometry())

            s2lastyear = s2lastyear.addBands([asap_highres.get_nObs_band(s2lastCol).clip(roi.geometry())])  # ***PG Edit***

        # Add the last years Sentinel-2 composite.
        s2lastyear_visualization = s2lastyear.getMapId(s2_visualization_options)

        s2ly_nb = int(s2lastyear.bandNames().size().getInfo())

        if s2ly_nb > 0:
            layers.append({
                'mapid': s2lastyear_visualization['mapid'],
                'label': "Sentinel-2 ({} to {})".format(ee.Date.format(start_date_1Y, 'd MMM Y').getInfo(),
                                                        ee.Date.format(end_date_1Y, 'd MMM Y').getInfo()),
                'token': s2lastyear_visualization['token']
            })

        # ***PG Edit***
        if do_add_DOY_band:# prepare inputs for visualization of observation DOY band

            # currently get min and max DOY from daterange
            minmax_doy_1 = [start_date.getRelative('day', 'year').add(1).getInfo(), end_date.getRelative('day', 'year').add(1).getInfo()]
            minmax_doy_2 = [start_date_1Y.getRelative('day', 'year').add(1).getInfo(),end_date_1Y.getRelative('day', 'year').add(1).getInfo()]

            s2_DOY_visopts1 = {'bands': 'DOY', 'min': str(minmax_doy_1[0]), 'max': str(minmax_doy_1[1]), 'palette': "08ff1d, ff1306"}
            s2_DOY_visopts2 = {'bands': 'DOY', 'min': str(minmax_doy_2[0]), 'max': str(minmax_doy_2[1]), 'palette': "08ff1d, ff1306"}
            # compute visualization:
            s2now_DOY_vis = s2now.getMapId(s2_DOY_visopts1)
            s2last_DOY_vis = s2lastyear.getMapId(s2_DOY_visopts2)
            # compile layer specific information
            doy_layer_info_this = {'mapid': s2now_DOY_vis['mapid'],
                                   'label':'S2 Observation DOY (min={0},max={1})'.format(minmax_doy_1[0],minmax_doy_1[1]),
                                   'token': s2now_DOY_vis['token']}

            doy_layer_info_last = {'mapid': s2last_DOY_vis['mapid'],
                                   'label':'S2 Observation DOY (min={0},max={1})'.format(minmax_doy_2[0],minmax_doy_2[1]),
                                   'token': s2last_DOY_vis['token']}
            # append specific information to result list
            layers.append(doy_layer_info_this)
            layers.append(doy_layer_info_last)

        # ***PG Edit***
        if do_add_nOBS_band:  # prepare inputs for visualization of Number of Observations (nObs)_band:

            # currently get min max values for stretch as [0, nDaysInterval]
            minmax_nObs_1 = [0, minmax_doy_1[1] - minmax_doy_1[0]]
            minmax_nObs_2 = [0, minmax_doy_2[1] - minmax_doy_2[0]]

            s2_NOBS_visopts1 = {'bands': 'nObs', 'min': str(minmax_nObs_1[0]), 'max': str(minmax_nObs_1[1]), 'palette': '000000, 2892C7, FAFA64, E81014'}
            s2_NOBS_visopts2 = {'bands': 'nObs', 'min': str(minmax_nObs_2[0]), 'max': str(minmax_nObs_2[1]), 'palette': '000000, 2892C7, FAFA64, E81014'}

            # compute visualization:
            s2now_NOBS_vis = s2now.getMapId(s2_NOBS_visopts1)
            s2last_NOBS_vis = s2lastyear.getMapId(s2_NOBS_visopts2)

            # compile layer specific information
            nObs_layer_info_this = {'mapid': s2now_NOBS_vis['mapid'],
                                   'label': 'S2 Number of clear observations (min={0},max={1})'.format(minmax_nObs_1[0], minmax_nObs_1[1]),
                                   'token': s2now_NOBS_vis['token']}

            nObs_layer_info_last = {'mapid': s2last_NOBS_vis['mapid'],
                                   'label': 'S2 Number of clear observations (min={0},max={1})'.format(minmax_nObs_2[0], minmax_nObs_2[1]),
                                   'token': s2last_NOBS_vis['token']}

            # append specific information to result list
            layers.append(nObs_layer_info_this)
            layers.append(nObs_layer_info_last)


        if (s2now_nb > 0) and (s2ly_nb > 0):
            s2diff = s2now.normalizedDifference(['B8', 'B4']).subtract(s2lastyear.normalizedDifference(['B8', 'B4']))

            red2green = ['ffffff', 'a50026', 'd73027', 'f46d43', 'fdae61', 'fee08b', 'ffffbf', 'd9ef8b', 'a6d96a',
                         '66bd63', '1a9850', '006837'];

            s2diff_visualization_options = {
                'palette': ','.join(red2green),
                'min': -0.6,
                'max': 0.6,
                'format': 'png'
            }

            s2diff_visualization = ee.Image(s2diff).getMapId(s2diff_visualization_options)

            layers.append({
                'mapid': s2diff_visualization['mapid'],
                'label': 'Sentinel-2 NDVI difference',
                'token': s2diff_visualization['token']
            })

        if use_asap_pbc is False:

            L8NowCol = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')\
                            .filterDate(start_date, end_date)\
                            .filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_pcent)\
                            .map(asap_highres.L8_TOA_masking)\
                            .map(asap_highres.add_DOY_band)\
                            .filterBounds(roi.geometry())

            l8now = ee.Image(L8NowCol.mosaic()).clip(roi.geometry())

            l8now = l8now.addBands([asap_highres.get_nObs_band(L8NowCol).clip(roi.geometry())])  # ***PG Edit***

        elif use_asap_pbc is True:

            l8now = asap_highres.get_single_asap_pbc_layer(start_date,
                                                           end_date,
                                                           sensor='L8',
                                                           region=roi.geometry(),
                                                           weights=weights,
                                                           add_DOY=do_add_DOY_band,
                                                           add_nOBS=do_add_nOBS_band)  # ***PG Edit***


        # A set of visualization parameters using the landcover palette.
        l8_visualization_options = {
            'bands': ','.join(['B5', 'B6', 'B4']),
            'min': 0,
            'max': ','.join(['0.6', '0.6', '0.4']),
            'format': 'png'
        }

        l8now_visualization = l8now.getMapId(l8_visualization_options)
        l8now_nb = int(l8now.bandNames().size().getInfo())

        if l8now_nb > 0:
            layers.append({
                'mapid': l8now_visualization['mapid'],
                'label': "Landsat-8 ({}  to {})".format(ee.Date.format(start_date, 'd MMM Y').getInfo(),
                                                        ee.Date.format(end_date, 'd MMM Y').getInfo()),
                'token': l8now_visualization['token']
            })

        if use_asap_pbc is False:

            # mosaic() based composite
            L8LastCol = ee.ImageCollection('LANDSAT/LC08/C01/T1_RT_TOA')\
                            .filterDate(start_date_1Y, end_date_1Y) \
                            .filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_pcent)\
                            .map(asap_highres.L8_TOA_masking)\
                            .map(asap_highres.add_DOY_band)\
                            .filterBounds(roi.geometry())

            l8lastyear = ee.Image(L8LastCol.mosaic()).clip(roi.geometry())

            l8lastyear = l8lastyear.addBands([asap_highres.get_nObs_band(L8LastCol).clip(roi.geometry())])  # ***PG Edit***

        elif use_asap_pbc is True:
            # ***PG Edit***
            l8lastyear = asap_highres.get_single_asap_pbc_layer(start_date_1Y,
                                                                end_date_1Y,
                                                                sensor='L8',
                                                                region=roi.geometry(),
                                                                weights=weights,
                                                                add_DOY=do_add_DOY_band,
                                                                add_nOBS=do_add_nOBS_band)

        l8lastyear_visualization = l8lastyear.getMapId(l8_visualization_options)

        l8ly_nb = int(l8lastyear.bandNames().size().getInfo())

        # ***PG Edit***
        if do_add_DOY_band:  # prepare inputs for visualization of observation DOY band

            # assuming minmax_doy_1 minmax_doy_2 has already been computed for s2
            # also assuming s2_DOY_visopts1 has already been computed for S2 - is also valid for L9

            # compute visualization:
            l8now_DOY_vis = s2now.getMapId(s2_DOY_visopts1)
            l8lastyear_DOY_vis = s2lastyear.getMapId(s2_DOY_visopts2)

            # compile layer specific information
            l8_doy_layer_info_this = {'mapid': l8now_DOY_vis['mapid'],
                                   'label': 'L8 Observation DOY (min={0},max={1})'.format(minmax_doy_1[0],
                                                                                       minmax_doy_1[1]),
                                   'token': l8now_DOY_vis['token']}

            l8_doy_layer_info_last = {'mapid': l8lastyear_DOY_vis['mapid'],
                                   'label': 'L8 Observation DOY (min={0},max={1})'.format(minmax_doy_2[0],
                                                                                       minmax_doy_2[1]),
                                   'token': l8lastyear_DOY_vis['token']}
            # append specific information to result list
            layers.append(l8_doy_layer_info_this)
            layers.append(l8_doy_layer_info_last)

        # ***PG Edit***
        if do_add_nOBS_band:  # prepare inputs for visualization of Number of Observations (nObs)_band:

            # compute visualization:
            l8now_NOBS_vis = l8now.getMapId(s2_NOBS_visopts1)
            l8lastyear_NOBS_vis = l8lastyear.getMapId(s2_NOBS_visopts2)

            # compile layer specific information
            l8_nObs_layer_info_this = {'mapid': l8now_NOBS_vis['mapid'],
                                    'label': 'L8 Number of clear observations (min={0},max={1})'.format(minmax_nObs_1[0],
                                                                                                     minmax_nObs_1[1]),
                                    'token': l8now_NOBS_vis['token']}

            l8_nObs_layer_info_last = {'mapid': l8lastyear_NOBS_vis['mapid'],
                                    'label': 'L8 Number of clear observations (min={0},max={1})'.format(minmax_nObs_2[0],
                                                                                                     minmax_nObs_2[1]),
                                    'token': l8lastyear_NOBS_vis['token']}

            # append specific information to result list
            layers.append(l8_nObs_layer_info_this)
            layers.append(l8_nObs_layer_info_last)

        if l8ly_nb > 0:
            layers.append({
                'mapid': l8lastyear_visualization['mapid'],
                'label': "Landsat-8 ({} to {})".format(ee.Date.format(start_date_1Y, 'd MMM Y').getInfo(),
                                                       ee.Date.format(end_date_1Y, 'd MMM Y').getInfo()),
                'token': l8lastyear_visualization['token']
            })

        if (l8now_nb > 0) and (l8ly_nb > 0):
            l8diff = l8now.normalizedDifference(['B5', 'B4']).subtract(l8lastyear.normalizedDifference(['B5', 'B4']))

            red2green = ['ffffff', 'a50026', 'd73027', 'f46d43', 'fdae61', 'fee08b', 'ffffbf', 'd9ef8b', 'a6d96a',
                         '66bd63', '1a9850', '006837'];

            l8diff_visualization_options = {
                'palette': ','.join(red2green),
                'min': -0.6,
                'max': 0.6,
                'format': 'png'
            }

            l8diff_visualization = l8diff.getMapId(l8diff_visualization_options)

            layers.append({
                'mapid': l8diff_visualization['mapid'],
                'label': 'Landsat 8 NDVI difference',
                'token': l8diff_visualization['token']
            })

        if not layers:
            layers.append({'mapid': None, 'label': 'No S2, L8 found for periods', 'token': None})
        else:
            background = ee.Image().getMapId({})
            layers.append({'mapid': background['mapid'], 'label': 'Google Background', 'token': background['token']})
            # The number of samples we want to use to train our classifier.

        self.response.out.write(json.dumps(layers))


app = webapp2.WSGIApplication([
    ('/', MainPage),
    ('/getRegion', AsapInput),
    ('/getmapdata', GetMapData)
], debug=True)

def main_locator(): pass
if __name__ == '__main__':

    #TODO: TBD cloud cover = 40, RGB band combination should come from AJAX

    request = {'asap1_id': '85', 'cloudpcent': '40', 'end_period': '2018-09-19', 'deltad': '21', 'use_asap_pbc':False} #Umbria
    # request = {'asap1_id': '262', 'cloudpcent': '10', 'end_period': '2018-07-31', 'deltad': '15'}

    test = AsapInput(request)
    # test.get()
    pbc_map = GetMapData(request).get()

    for info in pbc_map:
        URL = 'https://earthengine.googleapis.com/map/{0}/%7Bz%7D/%7Bx%7D/%7By%7D?token={1}'.format(info['mapid'],info['token'])
        print('***********************************************************************************')
        print(info['label'])
        print(URL)

    print('URL')