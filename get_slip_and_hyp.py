# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import SpecSyn2
from SpecSyn2 import *

import slip_from_specsyn
from slip_from_specsyn import *

import mags
from mags import *
import fmomentN
from fmomentN import *
import numpy as np
from scipy.stats import truncexpon
from scipy.optimize import fsolve
import TaperSlip
from TaperSlip import *
import Stat_Hypo
from Stat_Hypo import *

def get_slip_and_hyp(Mw,mech,dim,corr,samp,acf,seed,taper_width,taper_window):
    W = dim[0]
    L = dim[1]

    corr = [corr[0]*1000, corr[1]*1000, corr[2] ]
    
    lx = np.linspace(0,L,num=int(L/samp[1])+1)
    lz = np.linspace(0,W,num=int(W/samp[0])+1)

    samp2 = [samp[0]*1000, samp[1]*1000,]


    if np.remainder(lx.shape,2)==1 and np.remainder(lz.shape,2)==1:
        L1 = L
        W1 = W
        G,spar = SpecSyn2([W1*1000, L1*1000],samp2,corr,acf,seed)
    elif np.remainder(lx.shape,2)==0 and np.remainder(lz.shape,2)==1:
        L1 = L + samp[1]
        W1 = W
        G,spar = SpecSyn2([W1*1000, L1*1000],samp2,corr,acf,seed)
        G = G[:,:-1]
    elif np.remainder(lx.shape,2)==1 and np.remainder(lz.shape,2)==0:
        L1 = L 
        W1 = W + samp[0]
        G,spar = SpecSyn2([W1*1000, L1*1000],samp2,corr,acf,seed)
        G = G[:-1,:]
    elif np.remainder(lx.shape,2)==0 and np.remainder(lz.shape,2)==0:
        L1 = L + samp[1]
        W1 = W + samp[0]
        G,spar = SpecSyn2([W1*1000, L1*1000],samp2,corr,acf,seed)
        G = G[:-1,:-1]

    #G,spar = SpecSyn2([W*1000, L*1000],samp,corr,acf,seed)
    #samp = [samp[0]/1000, samp[1]/1000,]
    slip, slip_taper = slip_from_specsyn(G,[W,L],Mw,samp,taper_width,taper_window)

    lx = np.linspace(0,L,slip.shape[1])
    lz = np.linspace(0,W,slip.shape[0])

    slip_pos, hypo_x, hypo_z = Stat_Hypo(slip_taper,lx,lz,mech)


    return slip, slip_taper, hypo_x, hypo_z, slip_pos


