# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import mags
from mags import *
import fmomentN
from fmomentN import *
import numpy as np
from scipy.stats import truncexpon
from scipy.optimize import fsolve
import TaperSlip
from TaperSlip import *

def slip_from_specsyn(G,dim,Mw,samp,taper_width=[1,1,1],taper_function='hn'):
    
    # get slip from specsyn
    W = dim[0]  # Needed in km
    L = dim[1]
    # lift field above zero; % simple water-level
    Y = G - np.min(G)

    # rescale slip distribution to match desried/computed moment
    mo,mw = fmomentN(Y,[W,L])
    Mo = fMw2MoN(Mw)
    slip = Y*(Mo/mo)

    # Changing from Normal to Truncated distribution  
    slip_mean = np.mean(slip)  # mean slip
    slip_max = 10**(0.95*np.log10(slip_mean) + 0.62)   # umax; # FOR SS, using Thingbaijam et al., (2015)

    ep = lambda uc: slip_mean - truncexpon.mean(b=slip_max / uc, scale=uc)
    uc = fsolve(ep, 0.1 * slip_mean)[0]

    slip[np.unravel_index(np.argsort(slip.flatten()), slip.shape)] = np.sort(
            truncexpon.rvs(b=slip_max / uc, scale=uc, size=slip.size))

    # Taper the slip here now and rescale to desired moment
    # Find Taper in terms of number of points
    ntx = round(taper_width[0]/samp[1]) 
    ntt = round(taper_width[1]/samp[0])
    ntb = round(taper_width[2]/samp[0])

    slip_taper = TaperSlip(slip,[ntx,ntt,ntb],taper_function)

    # Rescaling now
    mo,mw = fmomentN(slip_taper,[W,L])
    Mo = fMw2MoN(Mw)
    slip_taper = slip_taper*(Mo/mo)

    return slip,slip_taper



                      
        