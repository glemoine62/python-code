import sys
# Configure if snappy not in python dist-packages
sys.path.append('/home/lemoen/.snap/snap-python')

import snappy

from snappy import ProductIO, GPF, HashMap

file1 = sys.argv[1]

print "Reading {}".format(file1)

infile = ProductIO.readProduct(file1)

parameters = HashMap()

# Apply-Orbit-File with default parameters
aof = GPF.createProduct("Apply-Orbit-File", parameters, infile)
# Remove-GRD-Border-Noise with default parameters

parameters.put('borderLimit', '500')
parameters.put('trimThreshold', '0.5')
rbn = GPF.createProduct("Remove-GRD-Border-Noise", parameters, aof)

# Calibration to Gamma for both polarizations
parameters = HashMap()

#parameters.put('createGammaBand', 'true')
parameters.put('selectedPolarisations', 'VV,VH')
parameters.put('auxFile', 'Product Auxiliary File')
parameters.put('outputGammaBand', 'true')
parameters.put('outputSigmaBand', 'false')

calib = GPF.createProduct("Calibration", parameters, rbn)

# RemoveThermalNoise
parameters = HashMap()
parameters.put('selectedPolarisations', 'Gamma0_VV,Gamma0_VH')
#parameters.put('removeThermalNoise', 'true')
# parameters.put('reIntroduceThermalNoise', 'false')
tnr = GPF.createProduct("ThermalNoiseRemoval", parameters, calib)

# Terrain-Correction to 10 m pixel spacing in Auto-UTM
parameters = HashMap()
parameters.put('demName', 'SRTM 1Sec HGT')
parameters.put('demResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
parameters.put('pixelSpacingInMeter', '10.0')
parameters.put('sourceBandNames', 'Gamma0_VV,Gamma0_VH')
parameters.put('mapProjection', 'AUTO:42001')
parameters.put('nodataValueAtSea', 'false')

target = GPF.createProduct("Terrain-Correction", parameters, calib)

outfile = sys.argv[2]

ProductIO.writeProduct(target, outfile, "BEAM-DIMAP")
print("Done.")