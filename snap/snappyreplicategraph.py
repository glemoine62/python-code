import sys
# Configure if snappy not in python dist-packages
sys.path.append('/home/lemoen/.snap/snap-python')

import snappy

from snappy import ProductIO, GPF, HashMap

file1 = sys.argv[1]

print("Reading {}".format(file1)

infile = ProductIO.readProduct(file1)

parameters = HashMap()

# Apply-Orbit-File with default parameters
aof = GPF.createProduct("Apply-Orbit-File", parameters, infile)
# Remove-GRD-Border-Noise with default parameters
rbn = GPF.createProduct("Remove-GRD-Border-Noise", parameters, aof)

# Calibration to Gamma for both polarizations
parameters.put('outputGammaBand', True)
parameters.put('selectedPolarisations', 'VV,VH')

calib = GPF.createProduct("Calibration", parameters, aof)

# RemoveThermalNoise
parameters = HashMap()
parameters.put('selectedPolarisations', 'VV,VH')
parameters.put('removeThermalNoise', True)
parameters.put('reIntroduceThermalNoise', False)
rtn = GPF.createProduct("RemoveThermalNoise", parameters, calib)

# Terrain-Correction to 10 m pixel spacing in Auto-UTM
parameters = HashMap()
parameters.put('demName', 'SRTM 1Sec HGT')
parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('pixelSpacingInMeter', 10.0)
parameters.put('sourceBands', 'Gamma0_VV,Gamma0_VH')
parameters.put('mapProjection', 'AUTO:42001')
parameters.put('nodataValueAtSea', False)

target = GPF.createProduct("Terrain-Correction", parameters, calib)

outfile = "/home/lemoen/Downloads/S1A_S1_ApplyOrbit_Calibrate_ML_TC"

ProductIO.writeProduct(target, outfile, "BEAM-DIMAP")
print("Done.")