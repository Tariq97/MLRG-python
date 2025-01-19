# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, Dec 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import math
import numpy as np


#::::::::::::::
#fMo2MwN.py
#::::::::::::::

#function [Mw] = fMo2MwN(Mo)
#
#	function [Mw] = fMo2MwN(Mo) calculates the
#	magnitude Mw for a given moment in Nm
#	
#	for reference, see Thorne & Lay, page 384
#
#	mmai, 02/08/98
#	--------------

def fMo2MwN(Mo):
    Mw = (2.0/3.0)*(math.log10(Mo) - 9.05)
    return Mw


#::::::::::::::
#fMo2Mw.py
#::::::::::::::

#function [Mw] = fMo2Mw(Mo)
#
#	function [Mw] = fMo2Mw(Mo) calculates the
#	magnitude Mw for a given moment.
#	
#	for reference, see Thorne & Lay, page 384
#
#	mmai, 02/08/98
#	--------------

def fMo2Mw(Mo):
    fMo2Mw = (2.0/3.0)*(math.log10(Mo) - 16.05)
    return fMo2Mw


#::::::::::::::
#fMw2MoN.py
#::::::::::::::

#function [Mo] = fMw2MoN(Mw)
#
#	function [Mo] = fMw2MoN(Mw) calculates the
#	moment for a given magnitude [in Nm]
#
#	for reference see Thorne & Lay, page 384
#
#	mmai, 02/08/98
#	--------------

def fMw2MoN(Mw):
    fMw2MoN = 10**(1.5*Mw + 9.05)
    return fMw2MoN


#::::::::::::::
#fMw2Mo.py
#::::::::::::::

#function [Mo] = fMw2MoN(Mw)
#
#	function [Mo] = fMw2MoN(Mw) calculates the
#	moment for a given magnitude [in Nm]
#
#	for reference see Thorne & Lay, page 384
#
#	mmai, 02/08/98
#	--------------

def fMw2Mo(Mw):
    fMw2Mo = 10**(1.5*Mw + 16.05)
    return fMw2Mo




def nexp2(d):
    d=abs(d)
    if d <=1:
        xs=0
    else:
        if type(np.log2(d)) is not int:
            xs=np.ceil(np.log2(d))
        else:
            xs=np.log2(d)
    return xs

