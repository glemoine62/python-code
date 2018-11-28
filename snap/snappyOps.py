import sys
sys.path.append('/home/lemoen/.snap/snap-python')
from snappy import jpy
from snappy import ProductIO
from snappy import GPF
from snappy import HashMap

def get_snap_parameters(operator):
    
    op_spi = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi(operator)

    op_params = op_spi.getOperatorDescriptor().getParameterDescriptors()

    return op_params

operator = sys.argv[1]

op_params = get_snap_parameters(operator)

parameters = HashMap()

for param in op_params:
    print(param.getName(), param.getDefaultValue())
    
    parameters.put(param.getName(), param.getDefaultValue())

