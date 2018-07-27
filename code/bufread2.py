# bufread.py - reads a one line GeoJSON file (produced by mapillary) in buffered blocks of 1K
#              and parses point geometries and their attributes to a postgresql/postgis insert
#              statement in blocks of 500000 records.
#
# Version: 1.2  -  2018-07-26
#
# Author: Guido Lemoine - EC-JRC
#

import sys, io
import json
from StringIO import StringIO

srcfile = 'junk.geojson'
srcfile = 'mapillary-public.geojson'
outfile = 'mapillary_{}.sql'


start = '{"geometry"'
end = '"type":"Feature"}'
BUFSIZE=1024

f = io.open(srcfile, 'r', buffering=BUFSIZE)
buf = f.read(BUFSIZE)
rest = ''
noread = False

def parser(s, start, end):
  sidx = s.find(start)
  eidx = s.find(end)
  #print sidx, eidx
  if (sidx != -1) and (eidx > sidx):
  	#print "PARSE:", str(s[sidx:eidx + len(end)])
  	d = json.load(StringIO(str(s[sidx:eidx + len(end)])))
  	rest = str(s[eidx + len(end) + 1:])
  	#print "POSTREST: ", rest
  else:
  	print "MAJOR FUCKUP!!!!!!"
  	print "FUBAR: ", s
  	sys.exit(1)

  return d, rest

def json2sql(d):
	sql = """
	  insert into mapillary (cca, orientation, cfocal, camera_mode, calt, captured_at, created_at, 
	  user_key, compass_accuracy, sequence, gps_accuracy, width, fmm35, model, make, ca, key, height,
	  file_size, captured_with, gpano, wkb_geometry) values 
	  ({}, {}, {}, {}, {}, {}, {}, '{}', {}, '{}', {}, {}, {}, '{}', '{}', {}, '{}', {}, {}, '{}', '{}', ST_GeomFromText('{}', 4326));
	"""	
	coords = d['geometry']['coordinates']
	props = d['properties']
	if 'gpano' in props:
		gpano = 'Y' # props['gpano']
	else:
		gpano = 'N'
	if 'captured_with' in props:
		captured_with = props['captured_with']['app_version']
	else:
		captured_with = ''

	if 'captured_at' in props:
		captured_at = props['captured_at']
	else:
		captured_at = 0


	if 'model' in props:
		model = props['model']
	else:
		model = ''

	if 'make' in props:
		make = props['make']
	else:
		make = ''

	if 'cca' in props:
		cca = props['cca']
	else:
		cca = 0

	if 'cfocal' in props:
		cfocal = props['cfocal']
	else:
		cfocal = 0

	if 'calt' in props:
		calt = props['calt']
	else:
		calt = 0

	if 'file_size' in props:
		file_size = props['file_size']
	else:
		file_size = 0

	if 'altitude' in props:
		altitude = props['altitude']
	else:
		altitude = 0

	if 'sequence' in props:
		sequence = props['sequence']
	else:
		sequence = ''

	wkt = 'POINT({} {} {})'.format(coords[0], coords[1], altitude)
 	return sql.format(cca, props['orientation'], cfocal, props['camera_mode'], calt,
 		captured_at, props['created_at'], props['user_key'], props['compass_accuracy'], sequence,
 		props['gps_accuracy'], props['width'], props['fmm35'], model, make,
 		props['ca'], props['key'], props['height'], file_size, captured_with, gpano, wkt)

i = 0

fw = open(outfile.format(i), "w")
fw.write('BEGIN;')  # Switch-off autocommit for faster inserts

while len(buf) == BUFSIZE:
  #print "REST:", rest
  #print "BUF:", buf
  if len(rest) > BUFSIZE:
  	s = rest
  	noread = True
  else:
  	s = rest + str(buf)
  	noread = False
  #print "STRING:", s
  d, rest = parser(s, start, end)

  fw.write(json2sql(d))

  if not noread:
  	buf = f.read(BUFSIZE)

  i+=1
  if i % 500000 == 0:
  	fw.write('COMMIT;')
  	fw.flush()
  	fw.close()
  	fw = open(outfile.format(i), "w")
  	fw.write('BEGIN;')
  	print i

# Either one or 2 records are left in the remainder
# print "END: ", i, rest + str(buf)
if len(rest) > 10:
	d, rest = parser(rest + str(buf), start, end)
	fw.write(json2sql(d))

# If non-trivial rest is left, parse again
if len(rest) > 10:
	d, rest = parser(rest, start, end)
	fw.write(json2sql(d))

fw.write('COMMIT;')
fw.flush()
fw.close()
print i
print rest
