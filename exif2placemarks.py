#!/usr/bin/python2.7
import sys
import glob
import xml.dom.minidom
import exifread
import math

from osgeo import ogr
from osgeo import osr

def GetFile(file_name):
  """Handles opening the file.

  Args:
    file_name: the name of the file to get

  Returns:
    A file
  """

  the_file = None

  try:
    the_file = open(file_name, 'rb')
    
  except IOError:
    the_file = None
    
  return the_file


def GetHeaders(the_file):
  """Handles getting the EXIF headers and returns them as a dict.

  Args:
    the_file: A file object

  Returns:
    a dict mapping keys corresponding to the EXIF headers of a file.
  """

  data = exifread.process_file(the_file, 'UNDEF', False, False, False)
  return data


def DmsToDecimal(degree_num, degree_den, minute_num, minute_den,
                 second_num, second_den):
  """Converts the Degree/Minute/Second formatted GPS data to decimal degrees.

  Args:
    degree_num: The numerator of the degree object.
    degree_den: The denominator of the degree object.
    minute_num: The numerator of the minute object.
    minute_den: The denominator of the minute object.
    second_num: The numerator of the second object.
    second_den: The denominator of the second object.

  Returns:
    A deciminal degree.
  """

  degree = float(degree_num)/float(degree_den)
  minute = float(minute_num)/float(minute_den)/60
  second = float(second_num)/float(second_den)/3600
  return degree + minute + second


def GetGps(data):
  """Parses out the GPS coordinates from the file.

  Args:
    data: A dict object representing the EXIF headers of the photo.

  Returns:
    A tuple representing the latitude, longitude, and altitude of the photo.
  """

  lat_dms = data['GPS GPSLatitude'].values
  long_dms = data['GPS GPSLongitude'].values
  latitude = DmsToDecimal(lat_dms[0].num, lat_dms[0].den,
                          lat_dms[1].num, lat_dms[1].den,
                          lat_dms[2].num, lat_dms[2].den)
  longitude = DmsToDecimal(long_dms[0].num, long_dms[0].den,
                           long_dms[1].num, long_dms[1].den,
                           long_dms[2].num, long_dms[2].den)
  if data['GPS GPSLatitudeRef'].printable == 'S': latitude *= -1
  if data['GPS GPSLongitudeRef'].printable == 'W': longitude *= -1
  altitude = None
  direction = None

  try:
    alt = data['GPS GPSAltitude'].values[0]
    altitude = alt.num/alt.den
    if data['GPS GPSAltitudeRef'] == 1: altitude *= -1

  except KeyError:
    altitude = 0
  
  try:
    direction = data['GPS GPSImgDirection'].values[0]
    direction = float(direction.num/direction.den)

  except KeyError:
    direction = 0
    
  return latitude, longitude, altitude, direction


def CreateKmlDoc():
  """Creates a KML document."""

  kml_doc = xml.dom.minidom.Document()
  kml_element = kml_doc.createElementNS('http://www.opengis.net/kml/2.2', 'kml')
  kml_element.setAttribute('xmlns', 'http://www.opengis.net/kml/2.2')
  kml_element = kml_doc.appendChild(kml_element)
  document = kml_doc.createElement('Document')
  kml_element.appendChild(document)
  return kml_doc
  

def CreatePlacemark(kml_doc, file_name, the_file:
  """Creates a Placemark element in the kml_doc element.

  Args:
    kml_doc: An XML document object.
    file_name: The name of the file.
    the_file: The file object.
 
  Returns:
    An XML element representing the PhotoOverlay.
  """

  data = GetHeaders(the_file)
  coords = GetGps(data)

  po1 = None

  po1 = kml_doc.createElement('Placemark')
  po1.setAttribute('id', file_name)
  name = kml_doc.createElement('name')
  name.appendChild(kml_doc.createTextNode(file_name))
  description = kml_doc.createElement('description')

  description.appendChild(kml_doc.createTextNode(the_file.strip()))
      
  po1.appendChild(name)
  po1.appendChild(description)
  
  point = kml_doc.createElement('Point')
  coordinates = kml_doc.createElement('coordinates')
  coordinates.appendChild(kml_doc.createTextNode('%s,%s,%s'.format(coords[1], coords[0], coords[2])))
  point.appendChild(coordinates)
  po1.appendChild(point)
  document = kml_doc.getElementsByTagName('Document')[0]
  document.appendChild(po1)


def CreateKmlFile(flist):
  files = {}
  for f in flist:
    the_file = GetFile(f)
    if the_file is None:
      print "{} is unreadable".format(fname)
    else:
      files[f] = the_file
  
  kml_doc = CreateKmlDoc()
  
  for key in sorted(files.iterkeys()):
    CreatePlacemark(kml_doc, key, files[key])
    

  kml_file = open("{}_pm.kml".format(dir_name), 'w')
  kml_file.write(kml_doc.toprettyxml('  ', newl='\n', encoding='utf-8'))
  
  
def main():
  flist = glob.glob(sys.argv[1] + '/*.JPG')
  CreateKmlFile(flist)

   
if __name__ == '__main__':
  main()
