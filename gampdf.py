# Created by Hugo Cruz Jimenez, August 2011, KAUST
#from scipy.special import *
from scipy import special 
import numpy
import math 

def gampdf(x,a,b):

#    print x
#    xx=len(x)
    if type(x)==numpy.ndarray:
        xx=len(x)
        iz=numpy.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=(1./(b**a*special.gamma(a)))*(1**(a-1))*numpy.exp(-(1/b))
#                iz[0]=(1./(b**a*special.gamma(a)))*(1**(a-1))*numpy.exp(-(1/b))
            else:
                iz[i]=(1./(b**a*special.gamma(a)))*(x[i]**(a-1))*numpy.exp(-(x[i]/b))
#                iz[i]=(1./(b**a*special.gamma(a)))*(x[i]**(a-1))*numpy.exp(-(x[i]/b))
#                return iz
#                return iz[i]

    else:
        iz=numpy.zeros(1)
        iz=(1./(b**a*special.gamma(a)))*(x**(a-1))*numpy.exp(-(x/b))
#        iz[0]=(1./(b**a*special.gamma(a)))*(x**(a-1))*numpy.exp(-(x/b))

    return iz
