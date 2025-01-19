# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np
import scipy.signal as ss
# from hann import *
from scipy.signal.windows import hann 

def TaperSlip(S,N=[5,5,5],window='hn'):

#function [S] = TaperSlip(S,N,window)
#
# function [S] = TaperSlip(S,N,'window')
# tapers the slip amplitudes at the fault boundaries 
# to avoid large slip at the edges. 
#	
# INPUT:
# S	- original grid
# N 	- vector of numbers of rows/cols to be tapered [left/right top bottom]
# window- 'hn' for Hanning window 
#	  'kw' for Kaiser window
#	  'tr' for triangular window
# 	  window can also be a vector [c1 c2 c3] with c(i) < 1; 
#	  in this case, the corresponding rows/cols will simply be 
#	  multiplied by that c(i)
#	  NOTE: this scheme only works if only one col/row is tapered
#
# Default values: N = [1 0 1], window = 'hn'
#
# OUTPUT: 
# S 	- tapered grid (same size as input grid)
#
# See also MATLAB function HANNING, and KAISER
	
# Written by Martin Mai (mmai@pangea.Stanford.EDU) 
# 05/08/98
# last change 08/26/99
# ------------------------------------------------
    
    
    # create taper window, i.e. Kaiser window (NOTE: increasing beta in 
    # Kaiser-Window widens main lobe and attenuates amplitude in the side lobes)
    # or hanning (cosine) from 0 to 1 in N steps on EACH side

    if isinstance(window,str):

        if window == 'hn':  # Added +2 as it is needed for this hann functions to match matlab hanning
            taperS = hann(2*N[0]+1 + 2) # for left/right columns 
            taperT = hann(2*N[1]+1 + 2) # for top row
            taperB = hann(2*N[2]+1 + 2) # for bottom rows 
            taperS = taperS[1:-1]
            taperT = taperT[1:-1]
            taperB = taperB[1:-1]

        elif window == 'kw':
    	    taperS = np.kaiser(2*N[0]+1,beta=6); taperT = [1] if N[1]==0 else np.kaiser(2*N[1]+1,beta=6); taperB = np.kaiser(2*N[2]+1,beta=6)
        
        elif window == 'tr':
            taperS = ss.triang(2*N[0]+1); taperT = ss.triang(2*N[1]+1); taperB = ss.triang(2*N[2]+1)	
        mm=len(taperB)
        winS=taperS[int(N[0]+1): int(2*N[0]+1) ]
#        print winS
        winT=taperT[0:int(N[1])]
#        print winT
        winB=taperB[mm-(int(N[2])-1)-1:mm]
#        print winB
    
    elif not isinstance(window,str):

#      i,j = find(N == 0)  # to make sure that rows/cols with N = 0 are not
        j=0
        while j < len(N):
            if N[j]==0:
                window[j]=1
            j += 1 

#      [i,j] = find(N == 0)  # to make sure that rows/cols with N = 0 are not
#      window[j] = 1	   # tapered in case s contains entries other than 1	

        winS = window[0]
        winT = window[1]
        winB = window[2]
    
    #%% check in case no taper is applied at one of the boundaries
    if len(winS) == 0: winS = 1
    if len(winT) == 0: winT = 1
    if len(winB) == 0: winB = 1
    
     
    #%% set-up bell-shaped tapering function 
    bell = np.ones((np.shape(S))) 
    j,k = np.shape(S)

#    print 'tipo de winS',type(winS)
#    print 'winS=',winS
#    print 'winT=',winT
#    print 'winB=',winB
    ls = len(winS) 
    lt = len(winT)
    lb = len(winB)

    for zs in range(0,j):
        bell[zs,0:ls] = bell[zs,0:ls]*winS[::-1]
#        print 'abcd',zs,k-ls, k
#        print 'abcd',zs,k-ls,k
        bell[zs,k-ls:k] = bell[zs,k-ls:k]*winS

    for zt in range(0,k):
        bell[0:lt,zt-1] = bell[0:lt,zt-1]*winT
        bell[j-lb:j,zt-1] = bell[j-lb:j,zt-1]*winB
    
    #%% finally, multiply input slip with bell-function
    S = S*bell
    return S
