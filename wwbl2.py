# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
def wblpdf(x,a,b):
    xm,xn=np.shape(x)
    x=x.reshape(1,xm*xn)
    iz=np.zeros(xm*xn)
    for i in range(0,xm*xn):
        iz=b*(a**(-b))*(x[i]**(b-1))*np.exp(-(x[i]/a)**b)

        return iz
