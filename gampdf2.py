# Created by Hugo Cruz Jimenez, August 2011, KAUST
#from scipy.special import *
from scipy import special 
import numpy
import math 

def gampdf(x,a,b):
    xm,xn=numpy.shape(x)
    x=x.reshape(1,xm*xn)
    iz=numpy.zeros(xm*xn)

    for i in range(0,xm*xn):
        iz=(1./(b**a*special.gamma(a)))*(x[i]**(a-1))*numpy.exp(-(x[i]/b))
        return iz
