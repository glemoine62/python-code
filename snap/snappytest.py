import sys
sys.path.append('/home/lemoen/.snap/snap-python')

import snappy

print "Hello, snappy!"

from snappy import ProductIO, GPF, jpy

file1 = sys.argv[1]

print("Reading...")
infile = ProductIO.readProduct(file1)

HashMap = jpy.get_type('java.util.HashMap')

parameters = HashMap()

aof = GPF.createProduct("Apply-Orbit-File", parameters, infile)

parameters.put('outputSigmaBand', True)
parameters.put('selectedPolarisations', 'VV,VH')

calib = GPF.createProduct("Calibration", parameters, aof)

parameters = HashMap()
parameters.put('nRgLooks', 5)
parameters.put('nAzLooks', 5)
parameters.put('outputIntensity', True)
parameters.put('grSquarePixel', True)

multilooked = GPF.createProduct("MultiLook", parameters, calib)

parameters = HashMap()
parameters.put('demName', 'SRTM 1Sec HGT')
#parameters.put('externalDEMNoDataValue', 0.0)
#parameters.put('externalDEMApplyEGM', True)
parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('pixelSpacingInMeter', 50.0)

parameters.put('sourceBands', 'Sigma0_VV,Sigma0_VH')

mapproj = """PROJCS['UTM Zone 32 / World Geodetic System 1984', 
  GEOGCS['World Geodetic System 1984', 
    DATUM['World Geodetic System 1984', 
      SPHEROID['WGS 84', 6378137.0, 298.257223563, AUTHORITY['EPSG',7030]], 
      AUTHORITY['EPSG',6326]], 
    PRIMEM['Greenwich', 0.0, AUTHORITY['EPSG',8901]], 
    UNIT['degree', 0.017453292519943295], 
    AXIS['Geodetic longitude', EAST], 
    AXIS['Geodetic latitude', NORTH]], 
  PROJECTION['Transverse_Mercator'], 
  PARAMETER['central_meridian', 9.0], 
  PARAMETER['latitude_of_origin', 0.0], 
  PARAMETER['scale_factor', 0.9996], 
  PARAMETER['false_easting', 500000.0], 
  PARAMETER['false_northing', 0.0], 
  UNIT['m', 1.0], 
  AXIS['Easting', EAST], 
  AXIS['Northing', NORTH]]"""

#parameters.put('mapProjection', mapproj)
#parameters.put('alignToStandardGrid', False)
parameters.put('nodataValueAtSea', False)
#parameters.put('saveDEM', False)
#parameters.put('saveLatLon', False)
#parameters.put('saveIncidenceAngleFromEllipsoid', False)
#parameters.put('saveLocalIncidenceAngle', False)
#parameters.put('saveProjectedLocalIncidenceAngle', False)
#parameters.put('saveSelectedSourceBand', True)
#parameters.put('outputComplex', False)
#parameters.put('applyRadiometricNormalization', False)
#parameters.put('saveSigmaNought', False)
#parameters.put('saveGammaNought', False)
#parameters.put('saveBetaNought', False)
#parameters.put('incidenceAngleForSigma0', 'Use projected local incidence angle from DEM')
#parameters.put('incidenceAngleForGamma0', 'Use projected local incidence angle from DEM')
#parameters.put('auxFile', 'Latest Auxiliary File')

target = GPF.createProduct("Terrain-Correction", parameters, calib)

outfile = "/home/lemoen/Downloads/S1A_S1_ApplyOrbit_Calibrate_ML_TC"

ProductIO.writeProduct(target, outfile, "GeoTIFF")


print("Done.")