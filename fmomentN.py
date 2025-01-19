# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, Dec 2024, KAUST

import numpy as np
import mags
from fMo2MwN import *

def fmomentN(ina,dim):
    #function [Mo,Mw] = fmomentN(in,dim)
    #
    # function [Mo,Mw] = fmomentN(in,dim) calculates the 
    # moment of an event for a given slip distribution (in cm)
    # if only IN is given, DIM = size(IN)
    #
    # INPUT:  in 	- 2D-array that constains slip values
    #	      dim	- source dimensions [W L]
    #
    # OUTPUT: Mo	- seismic moment, in Nm
    #	      Mw	- moment magnitude
    #
    # mmai, 02/08/98
    # --------------
   
    # if len(args)==1:
    #     ina=args[0]
    #     W,L = np.shape(ina) 
    # elif len(args)==2:
    #     ina=args[0]
    #     dim=args[1]
    #     W=dim[0]
    #     L=dim[1]
    W = dim[0]
    L = dim[1]
    
    mu = 3.3*1e10			# shear modulus = rigidity [N/m^2]
    s = np.mean(ina)/100.0		# average slip over fault surface [m]
    A = L*W*1e6;			# fault surface [m]
    
    Mo = mu*s*A			# moment [Nm]
    Mw = fMo2MwN(Mo)		# moment magnitude
    return Mo, Mw
