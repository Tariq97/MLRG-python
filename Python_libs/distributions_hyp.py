from scipy import special 
import numpy as np
import math 

# Created by Hugo Cruz Jimenez, August 2011, KAUST

def gampdf(x,a,b):
    if type(x)==np.ndarray:
        xx=len(x)
        iz=np.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=(1./(b**a*special.gamma(a)))*(1**(a-1))*np.exp(-(1/b))
            else:
                iz[i]=(1./(b**a*special.gamma(a)))*(x[i]**(a-1))*np.exp(-(x[i]/b))
    else:
        iz=np.zeros(1)
        iz=(1./(b**a*special.gamma(a)))*(x**(a-1))*np.exp(-(x/b))

    return iz

def gampdf2(x,a,b):
    xm,xn=np.shape(x)
    x=x.reshape(1,xm*xn)
    iz=np.zeros(xm*xn)

    for i in range(0,xm*xn):
        iz=(1./(b**a*special.gamma(a)))*(x[i]**(a-1))*np.exp(-(x[i]/b))
        return iz
    
def normpdf(x,a,b):

    if type(x)==np.ndarray:
        xx=len(x)
        iz=np.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=(1./(b*math.sqrt(2.0*np.pi)))*np.exp(  -(1-a)**2/(2.0*b**2) )
            else:
                iz[i]=(1./(b*math.sqrt(2.0*np.pi)))*np.exp(  -(x[i]-a)**2/(2.0*b**2) )

    else:
        iz=np.zeros(1)
        iz=(1./(b*math.sqrt(2.0*np.pi)))*np.exp(  -(x-a)**2/(2.0*b**2) )

    return iz

def wblpdf(x,a,b):

    if type(x)==np.ndarray:
        xx=len(x)
        iz=np.zeros(xx)
        for i in range(0,xx):
            if xx==1:
                iz=b*(a**(-b))*(1**(b-1))*np.exp(-(1/a)**b)
            else:
                iz[i]=b*(a**(-b))*(x[i]**(b-1))*np.exp(-(x[i]/a)**b)

    else:
        iz=np.zeros(1)
        iz=b*(a**(-b))*(x**(b-1))*np.exp(-(x/a)**b)

    return iz

def wwbl(x,a,b):

    if type(x)=='int':
        iz=np.zeros(1)
        if x==1:
            iz=b*(a**(-b))*(1**(b-1))*np.exp(-(1/a)**b)
        else:
            iz=b*(a**(-b))*(x**(b-1))*np.exp(-(x/a)**b)
    elif type(x)=='list':
        iz=np.zeros(len(x))
        for i in range(0,len(x)):
            iz[i]=b*(a**(-b))*(x[i]**(b-1))*np.exp(-(x[i]/a)**b)
    elif type(x)=='numpy.ndarray':
        print('***Si es arreglo 2D***')
        xm,xn=np.shape(x)
        x=x.reshape(1,xm*xn)
        iz=np.zeros((xm*xn))
        for i in range(0,xm*xn):
            iz[i]=b*(a**(-b))*(x[i]**(b-1))*np.exp(-(x[i]/a)**b)
    return iz

def wwbl2(x,a,b):
    xm,xn=np.shape(x)
    x=x.reshape(1,xm*xn)
    iz=np.zeros(xm*xn)
    for i in range(0,xm*xn):
        iz=b*(a**(-b))*(x[i]**(b-1))*np.exp(-(x[i]/a)**b)

        return iz