# Created by Hugo Cruz Jimenez, August 2011, KAUST
import numpy
def wblpdf(x,a,b):

#    print x
#    xx=len(x)
    if type(x)==numpy.ndarray:
        xx=len(x)
        iz=numpy.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=b*(a**(-b))*(1**(b-1))*numpy.exp(-(1/a)**b)
#                iz[0]=b*(a**(-b))*(1**(b-1))*numpy.exp(-(1/a)**b)
            else:
                iz[i]=b*(a**(-b))*(x[i]**(b-1))*numpy.exp(-(x[i]/a)**b)
#                iz[i]=b*(a**(-b))*(x[i]**(b-1))*numpy.exp(-(x[i]/a)**b)

    else:
        iz=numpy.zeros(1)
        iz=b*(a**(-b))*(x**(b-1))*numpy.exp(-(x/a)**b)
#        iz[0]=b*(a**(-b))*(x**(b-1))*numpy.exp(-(x/a)**b)

    return iz
