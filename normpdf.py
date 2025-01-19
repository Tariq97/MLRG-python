# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, Dec 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
from scipy import special 
import numpy
import math 

def normpdf(x,a,b):

    if type(x)==numpy.ndarray:
        xx=len(x)
        iz=numpy.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=(1./(b*math.sqrt(2.0*numpy.pi)))*numpy.exp(  -(1-a)**2/(2.0*b**2) )
            else:
                iz[i]=(1./(b*math.sqrt(2.0*numpy.pi)))*numpy.exp(  -(x[i]-a)**2/(2.0*b**2) )

    else:
        iz=numpy.zeros(1)
        iz=(1./(b*math.sqrt(2.0*numpy.pi)))*numpy.exp(  -(x-a)**2/(2.0*b**2) )

    return iz
