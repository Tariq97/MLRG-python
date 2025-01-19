# Created by Tariq Anwar Aquib,2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
import scipy.fftpack
import scipy.io

# hypocentre is index with z direction as first value and x direction as second value.

def compute_CL(slip,hyp,samp):


    # computing X and Y and crack length parameters:
    org_dist_x = (hyp[1] - 1)*samp[1]
    org_dist_y = (hyp[0] - 1)*samp[0]

    nz,nx = slip.shape

    lx = np.arange(0,samp[1]*nx,samp[1])
    lz = np.arange(0,samp[0]*nz,samp[0])

    lx = lx[:nx]
    lz = lz[:nz]

    L = lx[-1]
    W = lz[-1]

    x = np.linspace(-org_dist_x,L-org_dist_x,num=nx)
    y = np.linspace(org_dist_y,org_dist_y-W,num=nz)

    X,Y = np.meshgrid(x,y)

    l1 = np.max(X)
    l2 = np.abs(np.min(X))

    w1 = np.max(Y)
    w2 = np.abs(np.min(Y))

    X2 = np.zeros([nz,nx])
    Y2 = np.zeros([nz,nx])

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
            if X[i,j]>0:
                X2[i,j] = X[i,j]/l1
            else:
                X2[i,j] = X[i,j]/l2

    for i in range(0,Y.shape[0]):
        for j in range(0,Y.shape[1]):
            if Y[i,j]>0:
                Y2[i,j] = Y[i,j]/w1
            else:
                Y2[i,j] = Y[i,j]/w2

    # computing crack_length now
    crack_length = np.zeros([nz,nx])

    for j in range(0,len(y)):
        for i in range(0,len(x)):
            crack_length[j,i] = np.sqrt( x[i]**2 + y[j]**2)


    return X2, Y2, crack_length