# Created by Hugo Cruz Jimenez, August 2011, KAUST
import numpy as np
from farrprint import *
def WriteArray(*args):
#function WriteArray(A,dim,file)
#
# WriteArray(A,dim,file) simply writes the array A to
# the disk in the current directory under the name
# FILE; the sequence (for slip or rupture times) is
# that along strike (rows) are written as columns,
# down dip values (cols) are written as rows. DIM
# contains the dimensions [W L]
# 
# mmai, 05/17/2001
# ----------------

    A=args[0]
    dim=args[1]
    samp=args[2]
    filein=args[3]
    
    fid1 = open(filein, 'w')
    
    m,n = np.shape(A)
    W = dim[0]; L = dim[1]
#    x = np.linspace(1,L,n)
#    z = np.linspace(1,W,m)
    x = np.linspace(0,(n-1)*samp[1],n)
    z = np.linspace(0,(m-1)*samp[0],m)
##    x = np.linspace(1,(n-1)*samp[1],samp[1])
##    z = np.linspace(1,(m-1)*samp[0],samp[0])
#    x = np.linspace(1,samp[1],(n-1)*samp[1])
#    z = np.linspace(1,samp[0],(m-1)*samp[0])
    
    S = np.transpose(A)		## transpose for writing
    
    farrprint(fid1,[],'# -------')
    farrprint(fid1,[],'# File: {0}'.format(filein))
    farrprint(fid1,[],'# -------')
    farrprint(fid1,z,'# Ddco Z')
    farrprint(fid1,[],'# -------')
    farrprint(fid1,[],'# Strk X')
    farrprint(fid1,[],'# -------')

    for i in range(0,n):
        farrprint(fid1, np.concatenate((x(i),S[i][:]),1))
#        farrprint(fid1,[ i S[i][:] ])
      
    fid1.close()
