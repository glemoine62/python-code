#!/usr/bin/env python
"""ASAP prototype for high resolution view of alert area."""

import json
import os
import random
import datetime

import ee

#from google.appengine.api import urlfetch

import asap_compose


def GetMapData(request):
    """Handler for the layer selection."""

    gee_url = "https://earthengine.googleapis.com/map/MAPID/{z}/{x}/{y}?token=TOKEN"
    
    do_add_DOY_band = True  # ***PG Edit***
    do_add_nOBS_band = True  # ***PG Edit***

    """Sets up the request to Earth Engine and returns the map information."""
    ee.Initialize()
    ee.data.setDeadline(120000)
    #urlfetch.set_default_fetch_deadline(120000)

    gaul1 = ee.FeatureCollection('users/gglemoine62/asap_level1')
    asap1_id = int(request.get('asap1_id'))
    cloud_pcent = int(request.get('cloudpcent'))

    deltad = int(request.get('deltad'))
    end_date = ee.Date(request.get('end_period'))
    start_date = end_date.advance(-deltad, 'day')
    
    use_asap_pbc = request.get('use_asap_pbc')

    roi = ee.Feature(gaul1.filterMetadata('asap1_id', 'equals', asap1_id).first())

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

    if use_asap_pbc:
        weights = asap_compose.pbc_weights_selector(interval=deltad)

        s2now = asap_compose.get_single_asap_pbc_layer(start_date, end_date,
                                                       region=roi.geometry(), weights=weights,
                                                       add_DOY=do_add_DOY_band, add_nOBS=do_add_nOBS_band)# ***PG Edit***
    else:
        s2nowCol = ee.ImageCollection('COPERNICUS/S2')\
                        .filterDate(start_date, end_date)\
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent)\
                        .map(asap_compose.s2_cloudmasking)\
                        .map(asap_compose.add_DOY_band)\
                        .filterBounds(roi.geometry())

        s2now = ee.Image(s2nowCol.mosaic()).clip(roi.geometry())

        s2now = s2now.addBands([asap_compose.get_nObs_band(s2nowCol).clip(roi.geometry())])  # ***PG Edit***

    s2now_visualization = s2now.getMapId(s2_visualization_options)

    s2now_nb = int(s2now.bandNames().size().getInfo())
    # Add the recent Sentinel-2 composite.
    if s2now_nb > 0:
        layers.append({
            'tmsurl': gee_url.replace('MAPID', s2now_visualization['mapid']).replace('TOKEN', s2now_visualization['token']),
            'label': "Sentinel-2 ({} to {})".format(ee.Date.format(start_date, 'd MMM Y').getInfo(),
                                                    ee.Date.format(end_date, 'd MMM Y').getInfo())
        })


    if use_asap_pbc:

        s2lastyear = asap_compose.get_single_asap_pbc_layer(start_date_1Y, end_date_1Y,
                                                            region=roi.geometry(), weights=weights,
                                                            add_DOY=do_add_DOY_band, add_nOBS=do_add_nOBS_band)  # ***PG Edit***
    else:
        s2lastCol = ee.ImageCollection('COPERNICUS/S2')\
                        .filterDate(start_date_1Y, end_date_1Y)\
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_pcent)\
                        .map(asap_compose.s2_cloudmasking)\
                        .map(asap_compose.add_DOY_band)\
                        .filterBounds(roi.geometry())

        s2lastyear = ee.Image(s2lastCol.mosaic()).clip(roi.geometry())

        s2lastyear = s2lastyear.addBands([asap_compose.get_nObs_band(s2nowCol).clip(roi.geometry())])  # ***PG Edit*** 

    # Add the last years Sentinel-2 composite.
    s2lastyear_visualization = s2lastyear.getMapId(s2_visualization_options)

    s2ly_nb = int(s2lastyear.bandNames().size().getInfo())

    if s2ly_nb > 0:
        layers.append({
            'tmsurl': gee_url.replace('MAPID', s2lastyear_visualization['mapid']).replace('TOKEN', s2lastyear_visualization['token']),
            'label': "Sentinel-2 ({} to {})".format(ee.Date.format(start_date_1Y, 'd MMM Y').getInfo(),
                                                    ee.Date.format(end_date_1Y, 'd MMM Y').getInfo())
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
        doy_layer_info_this = {
                               'tmsurl': gee_url.replace('MAPID', s2now_DOY_vis['mapid']).replace('TOKEN', s2now_DOY_vis['token']),
                               'label':'S2 Observation DOY (min={0},max={1})'.format(minmax_doy_1[0],minmax_doy_1[1])
                               }

        doy_layer_info_last = {
                               'tmsurl': gee_url.replace('MAPID', s2last_DOY_vis['mapid']).replace('TOKEN', s2last_DOY_vis['token']),
                               'label':'S2 Observation DOY (min={0},max={1})'.format(minmax_doy_2[0],minmax_doy_2[1])
                               }
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
        nObs_layer_info_this = {
                                'tmsurl': gee_url.replace('MAPID', s2now_NOBS_vis['mapid']).replace('TOKEN', s2now_NOBS_vis['token']),
                                'label': 'S2 Number of clear observations (min={0},max={1})'.format(minmax_nObs_1[0], minmax_nObs_1[1])
                                }

        nObs_layer_info_last = {
                                'tmsurl': gee_url.replace('MAPID', s2last_NOBS_vis['mapid']).replace('TOKEN', s2last_NOBS_vis['token']),
                                'label': 'S2 Number of clear observations (min={0},max={1})'.format(minmax_nObs_2[0], minmax_nObs_2[1])
                                }

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
            'tmsurl': gee_url.replace('MAPID', s2diff_visualization['mapid']).replace('TOKEN', s2diff_visualization['token']),
            'label': 'Sentinel-2 NDVI difference'            
        })


    if not layers:
        layers.append({'mapid': None, 'label': 'No S2, L8 found for periods', 'token': None})
        
    return json.dumps(layers)


if __name__ == '__main__':

    #request = {'asap1_id': '85', 'cloudpcent': '10', 'end_period': '2018-09-19', 'deltad': '21'} #Umbria
    request = {'asap1_id': '262', 'cloudpcent': '10', 'end_period': '2018-07-31', 'deltad': '15', 'use_asap_pbc': False}

    pbc_map = GetMapData(request)

    print pbc_map

