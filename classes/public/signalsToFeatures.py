import numpy as np
from numpy import matlib

def signalsToFeatures(signals):

    Features = []      
    for var in signals:
        for i in [10,20,30,40,50,60,70,80,90]:
            Features.append(var + " " + str(i))
        Features.append(var + " mean")
        Features.append(var + " std")

        for i in [10,20,30,40,50,60,70,80,90]:
            Features.append(var + " " + str(i) + " deriv")
        Features.append(var + " mean deriv")
        Features.append(var + " std deriv")
    
    return Features   