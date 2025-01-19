# Created by Hugo Cruz Jimenez, August 2011, KAUST
# import commands
import os
import sys
import math
import numpy
import scipy

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
