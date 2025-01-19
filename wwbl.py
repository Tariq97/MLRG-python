# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy
def wblpdf(x,a,b):

    if type(x)=='int':
        iz=numpy.zeros(1)
        if x==1:
            iz=b*(a**(-b))*(1**(b-1))*numpy.exp(-(1/a)**b)
        else:
            iz=b*(a**(-b))*(x**(b-1))*numpy.exp(-(x/a)**b)
    elif type(x)=='list':
        iz=numpy.zeros(len(x))
        for i in range(0,len(x)):
            iz[i]=b*(a**(-b))*(x[i]**(b-1))*numpy.exp(-(x[i]/a)**b)
    elif type(x)=='numpy.ndarray':
        print('***Si es arreglo 2D***')
        xm,xn=numpy.shape(x)
        x=x.reshape(1,xm*xn)
        iz=numpy.zeros((xm*xn))
        for i in range(0,xm*xn):
            iz[i]=b*(a**(-b))*(x[i]**(b-1))*numpy.exp(-(x[i]/a)**b)
    return iz
