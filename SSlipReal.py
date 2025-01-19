# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import sys
from pprint import pprint
import pickle
#mimport matplotlib
#mimport matplotlib.cm as cm
#mimport matplotlib.mlab as mlab
#mimport matplotlib.pyplot as plt
#mfrom matplotlib.pyplot import figure, show
import fmomentN
import numpy as np
import math
#import plotslip2
from mags import *
from hann import *
from Calcs_all import *
#from CalcSigfromD import *
from SpecSyn2 import *
from TaperSlip import *
from WaterLevel import *
from fmomentN import *
from WriteArray import *

# [S,par] = SlipReal(srcpar,mech,acf,corr,seed,samp,grd,nexp,wlevel,taper,depth,dip,fig,outfile)
 
def SSlipReal(*args):
    #function [S,par] = SlipReal(srcpar,mech,acf,corr,seed,samp,grd,...
    # 			nexp,wlevel,taper,depth,dip,fig,outfile)
    #
    #  [S,par] = SlipReal4(srcpar,'mech','acf',corr,seed,samp,grd, ...
    #			nexp,wlevel,taper,depth,dip,outfile,fig) 
    #
    #    """ slip_real
    #    simulates a dislocation model for given source parameters and 
    #    source mechanism. If not given, source dimensions can be 
    #     computed from empirical scaling relations.
    #    The slip on the fault surface is calculated using the spectral 
    #    synthesis method in which the slip-amplitude spectrum is defined 
    #    through a spatial auto-correlation function or a power law decay;
    #    the phase is random in U(-pi,pi), but the entire field obeys
    #    Hermitian symmetry.
    #    The variables SRCPAR and MECH are required input; for all other 
    #    input parameters, an empty array {} will select the default values
    #    (see below).
    #  
    #    INPUT:
    #    srcpar - Array-structure with ALL source parameters OR 
    #             cell-array with source parameters and string to identify scaling;
    #             OR simple vector with source parameters
    #             For the last two options, srcpar can be of the form:
    #                  {Mw('rel'} --  source dimensions computed from scaling laws
    #                         rel ==('MB' uses Mai & Beroza (2000)
    #                         rel ==('WC' uses Wells & Coppersmith (1994)
    #                         rel ==('WG' uses USGS WorkingGroup99 M vs. A (2000)
    #                  {A('rel'}  --  area A (in km**2), rel is('MB' or('WC'
    #                  [W L]      --  Mw estimated from Wells & Coppersmith (1994),
    #                  [W L Mw]   --  D computed from given parameters
    #                  [W L Mw D] --  Mw will be scaled to match given D!
    #  
    #             NOTE: fault width W and length L are needed in km, mean slip D in m
    #  
    #              In case srcpar is given as an array-structure, make sure the naming
    #              in your array is EXAKT the same as needed in the code; the best idea
    #              is to run SlipReal, once with a standard set of input values, and then
    #              modify the entries in the output-structure PAR.
    #  
    #    mech   - faulting mechanism for the simulated event, needed to
    #                      compute source parameters from scaling relations
    #                     ('ss' or('SS' for strike-slipevents;  
    #                 ('ds' or('DS' for dip-slip events
    #                 ('al' or('AL' if both types should be considered
    #                          (may be useful in case of normal faulting events)
    #  
    #    acf    - string to denote autocorrelation function
    #                 ('ak' or('AK' for anisotropic von Karman ACF
    #                 ('ex' or('EX' for exponential ACF
    #                 ('fr' or('FR' for the fractal case (power law decay)
    #                 ('gs' or('GS' for Gaussian ACF
    #                      for this option YOU have to SPECIFY the correlation length
    #                      if {}, default('ak' is used
    #  
    #    corr   - correlation length and/or spectral decay parameters
    #                  [az ax] for Gauss or exponential ACF
    #                  [az ax H] for von Karman ACF H = Hurst number)
    #                  [D kc] for fractal slip where D is the fractal dimension;
    #                      kc: corner wavenumber beyond which the spectrum decays;
    #                      kc is related to the source dimensions, and is computed
    #                      as kc = 2*pi/(sqrt(L*W)) if it is not given
    #                  {} if corr is an empty matrix, the relevant parameters for the
    #                      given autocorrelation function will be computed
    #                      (NOT true for the Gaussian since no relations have been
    #                      established between corr length and source parameters)
    #  
    #    seed   - structural array of seed values for the random-number generator, 
    #                  called at various locations in the code; if seed == {}, then 
    #                  the code uses an expression like 
    #                         ('Rseed = sum(100*clock)' 
    #                         ('randn('seed', Rseed)' <-- uses MATLAB4 generators!!
    #                  to generate the random numbers; the value Rseed is stored in
    #                  the output structure par.
    #                  The sequence is as follows (also returned by the code):
    #                          seed.SS = SSseed; 1x2-array, used in SpecSyn2
    #                          seed.WL = WLseed; 1x1-array, used in WaterLevel
    #                          seed.CS = CSseed; 2x2-array, used in CalcSigfromD
    #                          seed.RC = RCseed; 3x2-array, used in CalcDimfromM
    #                          seed.RWC = RWCseed; 3x2-array, used in CalcDimWC
    #                          seed.CR = CRseed; 1x1, 4x2, 5x2, used in CalcCorrfromLW
    #                  Hence, you can run the code once, get the corresponding array 
    #                  and use it again to create the EXACT SAME random slip model.
    #  
    #    samp   - sampling of dislocation model in z,x direction [dz dx]
    #                  NOTE: the final sampling may be finer in case the option('pow2' 
    #                      is given as('y' (see SPECSYN2) or sampling must be adjusted for
    #                      the source dimensions
    #  
    #    grd	  - slip function to be defined on grid-nodes or subfaults 
    #                 ('nod' for grid-node definition [grid-size (L/dx+1)*(W/dz+1)] 
    #                 ('sub' for sub-fault definition [grid-size (L/dx) * (W/dz)]  
    #  
    #    nexp   - non-linear scaling exponent for the random field (i.e S = S**nexp) 
    #                  nexp < 1 smoothens the fields (steepens the spectrum) 
    #                  nexp == 1 doesn't change anything (default)
    #                  nexp > 1 roughens the field (flattens the spectrum)
    #                  the purpose of this variable is to allow for simulation of
    #                  slip distributions with large peak-slip values;
    #                  a good choice in this case is usually nexp = 1.25 - 1.5;
    #  
    #    wlevel - method to scale the zero-mean random field to nonnegative slip
    #                  wlevel == -4: field "lifted" above zero (default) 
    #                  wlevel ==  0: values < 0 will be set equal to zero 
    #                  wlevel == -1: values < 0 will be set to small random value
    #                  wlevel == -2: values < 0 will be set to 0.25*field value
    #                  wlevel == -3: values < mean-slip set to small random value
    #                  wlevel == scalar: values < scalar set to small random value
    #                  NOTE: wlevel = [] preserves the spectral decay; wlevel = -2
    #                  also preserves the slip spectrum rather well without intro-
    #                  ducing too much artificial high-wavenumbers, yet scales the
    #                  slip such that locally large-slip patches are present
    #  
    #    taper  - tapering the edges of the slip model
    #                 ('y' for default tapering of 2.5 km (i.e [2.5 2.5 2.5])
    #                  [left/right top bottom] (in km) for customized tapering
    #                  [left/right top bottom P] to apply an additional
    #                   depth-dependent tapering of the form z**P; 
    #                   P > 1 to obtain less slip in shallow part of the fault
    #  
    #    depth  - max. depth of rupture plane;
    #             option with depth range [zmin zmax] not implemented
    #                      zmin is the depth to the top of the fault plane
    #                      zmax is the maximum depth of fault plane (in km)  
    #                           (default: zmin = 0, zmax =15)  # I changed to 30 km
    #  
    #    dip	  - dip angle of fault plane (in degrees) (default = 90 deg)  
    #  
    #    fig	  - optional:('y' to view the slip realization; this will open
    #                  a figure window for each realization (set to('n' if called 
    #                  in a loop) (default:('y')
    #  
    #    outfile- optional: string for a filename to which the code writes 
    #                  the slip realization as a simple ascii 2D-array, where rows
    #                  correspond to depth (default:('n')
    #  
    #    OUTPUT:	
    #    S 	  - 2D-slip distribution
    #    par 	  - structure with all relevant source parameters 
    #  
    #    Type('type SlipReal4' to view the entire program and particularly the
    #    more detailed description of the program at the beginning.
     
    #    """
    
    # -------------------------
    # FIXED PARAMETERS
    # -------------------------
     
    #    # rigidity of crustal rocks
    mu = 3.3*1e10
       
    # taper window used; either('kw' for Kaiser window
    # (steep) or('hn' for smoother Hanning window
    tapwin = 'hn'
       
    # set to('y' if fields to be simulated on square-grids
    # of power-of-two size, and then resampled to desired
    # grid-size
    synnp2 = 'n'
       
       
    #### -----------------------------------------
    #### SET AND DISPLAY DEFAULT VALUES, DEPENDING
    #### ON WHAT IS GIVEN BY THE FUNCTION CALL
    #### -----------------------------------------
           
    if len(args)==0:
        print('                                                     ')
        print(' ----------------------------------------------------')     
        print(' Welcome to SlipReal, a tool to generate slip maps')
        print(' for finite-fault strong motion simulations.')
        print(' It seems you haven''t used SlipReal before, so here')
        print(' are some introductory comments on its use')
        print(' ----------------------------------------------------')
        print('')
        print(' A program like this of course requires various input')
        print(' variables (type(''help SlipReal'' to see all these),')
        print(' but you can get a head-start by just giving the source')
        print(' parameters fault width, length, and moment magnitude')
        print(' as srcpar = [W L M], and the faulting mechanism FM as')
        print('(''SS'' or(''DS'' for strike-slip or dip-slip earthquakes.')
        print(' Then type')
        print('          [S,par]=SlipReal(srcpar,FM) ')
        print('(')
        print(' and the code will run once, telling you which default')
        print(' values it has chosen, printlaying the resulting slip map')
        print(' and generating the output structure PAR that can then')
        print(' be used to run the code with varying input values')
       
    elif len(args)==1:
        srcpar=args[0]
        if type(srcpar) is dict:
            print('   Reading source parameters from given structural dictionary')
            outfile ='n';		fig ='y'
            mech = srcpar['mech'];  acf=srcpar['acf']
            corr = srcpar['corr'];	seed = srcpar['Rseed']
            samp = srcpar['samp'];	nexp = srcpar['nexp']
            grd = srcpar['grd']
            depth = srcpar['depth'];	dip  = srcpar['dip']
            wlevel = srcpar['wlevel']; taper = srcpar['taper']
            src = srcpar; 		srcpar = np.zeros(4) 
            srcpar[0] = src['dim'][0]; srcpar[1] = src['dim'][1]
            srcpar[2] = src['Mw'];	srcpar[3] = src['Dmean']/100.0
              
        else:
               print('   Need structural array as input or at least')
               print('   moment magnitude Mw and source mechanism')
               print('   Error  **** SlipReal4 stopped ****')
    elif len(args)==2:
       srcpar=args[0]
       mech=args[1]
       acf ='ak'; 	corr = []; 	seed =  []
       samp = [1, 1];	grd ='nod';	nexp = 1	
       wlevel = -4; 	taper ='y'; 	depth = 30
       dip = 90;	outfile ='y'; 	fig ='y';
       print('   Default values selected in SlipReal:') 	
       print('   ACF = AK; 	SAMP  = [1 1];	GRD = nod;	NEXP = 1')
       print('   WLEVEL = [];	TAPER = y;	DIP = 90 dg.; 	DEPTH  = 30 km')
    elif len(args)==3:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr = []; 	seed = [];	samp = [1, 1]
       grd ='nod';	nexp = 1;	wlevel = -4 
       taper ='y'; 	depth = 30; 	dip = 90
       outfile ='y'; 	fig ='y';
       print('   Corr. length: simulated')
       print('   Default values selected in SlipReal: 	')
       print('   SAMP  = [1 1];	GRD = nod;	NEXP = 1;  	WLEVEL = []')
       print('   TAPER = y;	DIP = 90 dg.; 	DEPTH  = 30 km(') 
    elif len(args)==4:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed = {}; 	samp = [1, 1];	grd ='nod';	nexp = 1
       wlevel = -4; 	taper ='y'; 	depth = 30; 	dip = 90
       outfile ='y'; 	fig ='y';
       print('   Default values selected in SlipReal:'	)
       print('   SAMP  = [1 1];	GRD = nod;	NEXP = 1;  	WLEVEL = []')
       print('   TAPER = y;	DIP = 90 dg.; 	DEPTH  = 30 km') 
    elif len(args)==5:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp = [1, 1];	grd ='nod';	nexp = 1
       wlevel = -4; 	taper ='y'; 	depth = 30; 	dip = 90 
       outfile ='y'; 	fig ='y';
       print('   Default values selected in SlipReal:')
       print('   SAMP  = [1 1];	GRD = nod;	NEXP = 1;  	WLEVEL = []')
       print('   TAPER = y;	DIP = 90 dg.; 	DEPTH  = 30 km' )
    elif len(args)==6:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd ='nod';	nexp = 1;	wlevel = -4
       taper ='y'; 	depth = 30; 	dip = 90 
       outfile ='n'; 	fig ='y';
       print('   Default values selected in SlipReal:')
       print('   GRD = nod;	NEXP = 1;  	WLEVEL = []')
       print('   TAPER = y;	DIP = 90 dg.; 	DEPTH  = 30 km')
    elif len(args)==7:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp = 1;	wlevel = -4; 	taper ='y'
       depth = 30; 	dip = 90
       outfile ='n'; 	fig='y';
       print('   SAMP and NEXP used as given')
       print('   Default values selected in SlipReal: 	')
       print('   --> NEXP = 1; WLEVEL = []; TAPER = y; DIP = 90 dg.; DEPTH  = 30 km')
    elif len(args)==8:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel = -4;	taper ='y'; 	depth = 30 
       dip = 90;	outfile ='n'; 	fig='y'
       print('   SAMP, NEXP, WLEVEL used as given')
       print('   Default values selected in SlipReal: 	')
       print('   --> WLEVEL = []; TAPER = y; DIP = 90 dg.; DEPTH  = 30 km') 
    elif len(args)==9:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper ='y';	depth = 30; 	dip = 90
       outfile ='n'; 	fig='y'
       print('   SAMP, NEXP, WLEVEL, used as given')
       print('   Default values selected in SlipReal:')
       print('   --> TAPER = y; DIP = 90 dg.; 	DEPTH  = 30 km')
    elif len(args)==10:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper=args[9]
       depth = 30;	dip = 90;	outfile ='n'; 	fig='y'
       print('   SAMP, NEXP, WLEVEL, TAPER used as given')
       print('   Default values selected in SlipReal: 	')
       print('   --> DEPTH = 30 km, DIP = 90 dg.')
    elif len(args)==11:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper=args[9]
       depth=args[10]
       dip = 90;	outfile ='n'; 	fig='y'
       print('   SAMP, NEXP, WLEVEL, TAPER, DEPTH used as given')
       print('   Default values selected in SlipReal: 	')
       print('   --> DIP = 90 dg.')
    elif len(args)==12:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper=args[9]
       depth=args[10]
       dip=args[11]
       outfile ='n';   fig='y'
    elif len(args)==13:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper=args[9]
       depth=args[10]
       dip=args[11]
       fig=args[12]
       outfile ='n'
    elif len(args)==14:
       srcpar=args[0]
       mech=args[1]
       acf=args[2]
       corr=args[3]
       seed=args[4]
       samp=args[5]
       grd=args[6]
       nexp=args[7]
       wlevel=args[8]
       taper=args[9]
       depth=args[10]
       dip=args[11]
       fig=args[12]
       outfile=args[13]


    #### ----------------------------------
    #### SOME SIMPLE CHECKS ON INPUT ERRORS
    #### ----------------------------------
       
    #### handle the case where acf is given and the corr length
    #### vector, but the length of corr doesn't comply with the
    #### given function
    if len(srcpar) == 0:
        print(' ----')
        print(' Come on ... - give me at least some source parameter ...')
        print(' ----')
        print(' Error  **** SlipReal4 stopped ****')
        return
      
    if acf =='ex' or acf =='EX' or acf =='gs' or acf =='GS':
        if len(corr) == 1 or len(corr) > 2 :
            print('   -- Number of elements given in vector CORR')
            print('      does not comply with given acf-function')
            print(' Error  **** SlipReal stopped ****')
            return
    elif acf =='ak' or acf =='AK' :
        if len(corr) == 1 or len(corr) == 2 or len(corr) > 3:
            print('   -- Number of elements given in vector CORR')
            print('      does not comply with given acf-function')
            print(' Error  **** SlipReal stopped ****')
            return
    elif acf =='fr' or acf =='FR':
        if (len(corr) >= 3):
            print('   -- Number of elements given in vector CORR')
            print('      does not comply with given acf-function')
            print(' Error  **** SlipReal stopped ****')
            return
        elif len(corr) == 1:
            print('   -- corner wavenumber kc not specified')
            print('      will be computed from source dimensions')
       
       
       
    #### -----------------------------------------------------------------
    #### SET RANDOM SEED ARRAYS
    #### the random seed numbers are stored in arrays to be able to later
    #### recreate the EXACT same slip realization; 
    #### set the initial seed values to empty arrays and have the 
    #### subroutines return the actual seed values into the structural 
    #### sub-array par.Rseed; OR
    #### retrieve seed-values from given structural array
    #### -----------------------------------------------------------------
#   if len(seed) == 0:
    print('@$%  este es seed ***', seed)

    if len(seed) == 0:
        RCseed = {} 
        RWCseed= {}
        CRseed = {}
        SSseed = {}
        CSseed = {}
        WLseed = {}
        print('   new SEED-values used')
    else:
        SSseed = seed['SS']
        WLseed = seed['WL']
        CSseed = seed['CS']
        RCseed = seed['RC']
        RWCseed = seed['RWC']
        CRseed = seed['CR']
        print('   SEED-values from previous simulation used')
        print('SSseed', SSseed)
        print('WLseed', WLseed)
        print('CSseed', CSseed)
        print('RCseed', RCseed)
        print('RWCseed', RWCseed)
        print('CRseed', CRseed)
       
      
    #### -----------------------------------------------------------------
    #### NON-LINEAR SCALING EXPONENT FOR SLIP
    #### the purpose of this variable, together with the('water level' 
    #### option, is to be able to "customize" the generated models, e.g. 
    #### create large peak slip values;
    #### nexp > 1 roughens the field (flatter spectrum)
    #### nexp < 1 smoothens the field (steeper spectrum)
    #### we generally run nexp = 1, but test showed nexp = 1.2-1.5 is a 
    #### reasonable good choice
    #### ------------------------------------
    #if isempty(nexp) == 1:

    if nexp == 0:
        nexp = 1
       
       
    #### ----------------------
    #### GET RUPTURE DIMENSIONS
    #### ----------------------
    #### computed from empirical scaling relations 
    #### or specified by the user, depending on option
    ####('MB or('WC' to use Mai & Beroza (2000)
    #### scaling relations or Wells & Coppersmith (1994)
       
    if type(srcpar) is list and type(srcpar[1])==str:
       
        rel = srcpar[1]
        par = srcpar[0] 
       
       
        #### distinguish cases of A or Mw based on value of par
        #### because there are no earthquakes with M > 10 !
        #    if length(par) > 1:
        if len(par) > 1:  # No me queda claro
            print('(')
            print(' Give either area A or magnitude M')
            print(' along with string for scaling relations')
            print(' Error *** SLIP REAL TERMINATED ***')
        else:
            if par < 10:
                Mw = par
                if rel =='WG':
                    print(' WorkingGroup Relations only for given area!')
                    print(' Error *** SLIP REAL TERMINATED ***')
                    L,W,D,Mw,RCseed = CalcDimfromM(Mw,mech,'d',rel,RCseed)
                else:
                    A = par
                    L,W,D,Mw,RCseed = CalcDimfromA(A,mech,rel,RCseed)
               
        if rel =='MB':
            print('   Source dimensions from Mai & Beroza (2000)')
            srcmeth ='empirical (M&B)'
        elif rel =='WC':
            print('   Source dimensions from Wells & Coppersmith (1994)')
            srcmeth ='empirical (W&C)'
         
             
    #### srcpar is NOT given as cell array     
    elif type(srcpar) is list and type(srcpar[1]) != str:
       
    #### if W,L are given, only Wells & Coppersmith (1994) relations are useful
        if len(srcpar) == 2:
            W = float(srcpar[0])
            L = float(srcpar[1])
            L,W,D,Mw,RWCseed = CalcDimDMWC(L,W,mech,RWCseed)
            srcmeth ='empirical (W&C)'
            print('   Moment Magnitude Mw from Wells & Coppersmith (1994)')
           
         
     #### handle cases in which fault length, width, mean slip and
     #### moment magnitude are given
        elif len(srcpar) == 3:
            W = float(srcpar[0])
            L = float(srcpar[1])
            Mw = float(srcpar[2])
            Mo = fMw2MoN(Mw)
            D = Mo/(mu*L*W*1e6)
            srcmeth ='specified'
            print('   Mean Slip D computed from given source parameters')
          
        elif len(srcpar) == 4:
            W = float(srcpar[0])
            L = float(srcpar[1])
            Mw = float(srcpar[2])
            D = float(srcpar[3])
            Mo = mu*(L*W*D*1e6)
            Mw = fMo2MwN(Mo)
            srcmeth ='specified'
            print('   Moment Magnitude Mw adjusted to reflect given Mean Slip D')
     
    #* #### check that maximum fault width does not exceed maximum allowable depth; 
    #* #### if that's case, the width will be adjusted to max allowable width
    maxW = depth/math.sin(np.pi*dip/180.0)
    if W > maxW:
        print('   -- computed fault width exceeds max. allowable width')
        print('      width will be adjusted to max. allowable width')
        W = math.floor(maxW)

    h = depth - W*math.sin(np.pi*dip/180.)      # top of fault plane
        
    #### check whether values of L and W, together with the selected
    #### spatial sampling, yield an integer number of point; otherwise
    #### the spectral synthesis will return an error message. For now
    #### we consider only one significant digit. The program will then
    #### adjust L and W in order to maintain the chosen sampling
    L = 0.1*round(L*10)		
    W = 0.1*round(W*10)
    if np.mod(L,samp[1]) != 0:
        print('   --> Need to adjust length L in order to be compatible')
        print('       with the chosen spatial sampling')
        nnx = L/samp[1]
        L  = round(nnx)*samp[1]
    if np.mod(W,samp[0]) != 0:
        print('   --> Need to adjust width W in order to be compatible')
        print('       with the chosen spatial sampling')
        nnz = W/samp[0] 
        W  = round(nnz)*samp[0]
    Mo = fMw2MoN(Mw)
    D = Mo/(L*W*1e6*mu)
        
     #### print(' final source parameters to screen')

    print('   ** Final Source Parameters:')
    print('  PARA-FIG    L = {0:.2f} km, W = {1:.2f} km, D = {2:.3f} m, Mw = {3:.2f}, Mo = {4:.3e} Nm'.format(L,W,D,Mw,Mo) )

    #### -----------------------------------------------------------------
    #### TAPERING OF SLIP FUNCTION, [left/right top bot wfct]
    #### to avoid large slip at the fault boundaries, we apply a taper 
    #### function that may vary for the left/right, top and bottom extent
    #### of the taper window; the default is set to 2.5 km. Additionally,
    #### a depth-dependent "weighting function" can be applied such as to 
    #### have larger slip values at depth
    #### -----------------------------------------------------------------

    print('EL TIPO DE TAPER ES: {0}'.format(type(taper)))
    print('ESTE ES TAPER: {0}'.format(taper))
    print('LONGITUD  DE TAPER ES: {0}'.format(len(taper)))


    if len(taper) == 4:
        twdpt = taper[3]
    elif len(taper) == 0:
        taper = [0, 0, 0]		# array of zeros in case of NO taper
           
    if taper =='y':		    	
        twkm = [2.5, 2.5, 2.5]	    	# set taper windows to default values 
    elif taper =='d':
        if'twfr' not in locals():
            twfr = 0.25
            twL = twfr*L; twW = twfr*W	# set taper window length to 25# of 
            twkm = [twL, twW, twW]		# the source size in each direction

    elif type(taper) is list and len(taper)>=3:	    	
        twkm = taper[0:3]		# set taper windows to given values   	
        print('******^^^^^^^^^^^^************* len(taper)={0}'.format(len(taper)) )	# defines the # of rows/cols of the
        print('******^^^^^^^^^^^^*************') 	# defines the # of rows/cols of the
        print(twkm)
        print('******^^^^^^^^^^^^*************') 	# defines the # of rows/cols of the
        print('******^^^^^^^^^^^^*************') 	# defines the # of rows/cols of the
   

        print('LONGITUD DE SAMP = {0}'.format(len(samp)))
       

        if len(samp) == 2: 
            ntx = round(twkm[0]/samp[1]) 
            print('*$ ntx {0}'.format(ntx))
            ntt = round(twkm[1]/samp[0])
            ntb = round(twkm[2]/samp[0])	 

#            print ('ntx, ntt, ntb',ntx, ntt, ntb) 	# defines the # of rows/cols of the
            print ('******^^^^^^^^^^^^*************') 	# defines the # of rows/cols of the
            print ('******^^^^^^^^^^^^*************') 	# defines the # of rows/cols of the
            print (ntx, ntt, ntb ,' defines the # of rows/cols of the')

            tapslp = [ntx, ntt, ntb] 	# defines the # of rows/cols of the
                                        ## output slip distribution to be tapered
            print('TAPSLP 1 {0}'.format(tapslp))
            print('TAPSLP 1', tapslp)

        elif len(samp) == 1: 
            ntx = round(twkm[0]/samp[0])  # Esta raro, deberia ser tambien 0 porque solo tiene 1, pero asi viene del de Martin
            print('##*$ ntx {0}'.format(ntx))
            ntt = round(twkm[1]/samp[0])
            ntb = round(twkm[2]/samp[0])

            print ('ntx, ntt, ntb',ntx, ntt, ntb) 	# defines the # of rows/cols of the
       
            tapslp = [ntx, ntt, ntb] 	# defines the # of rows/cols of the
                                        ## output slip distribution to be tapered
            print('TAPSLP 2 {0}'.format(tapslp))
            print('TAPSLP 2', tapslp)
   
       
       
    #### -----------------------------------------------
    #### PARAMETERS FOR AUTOCORRELATION FUNCTION OF SLIP
    #### -----------------------------------------------
       
    #### call function to compute correlation length/spectral decay for given Mw
    #### if vector with correlation length is empty
    if len(corr) == 0:
        corr,CRseed = CalcCorrfromLW([W, L, Mw],mech,acf,CRseed)
        corrmeth ='simulated'
        print('   Spectral decay parameters computed from empirical relations')
    ### check that the aspect ratio of the correlation length has the same "sign"
    ### as the fault dimensions
        if acf =='ak' or acf =='ex' or acf =='gs':
            if W > L and corr[0] < corr[1]: 
                corr[0:1] = [corr[1], corr[0]]
       
    elif len(corr) == 1 and acf =='fr' or acf =='FR': 
    #### need to compute corner wavenumber, based on source dimensions
        kc = 2*pi/(sqrt(L*W))
    #kc = 1/sqrt(L*W)
        corr = [corr[0], kc]
    else:
        corrmeth ='specified'
        print('   Spectral decay parameters used as specified')
      

#    return corr

    #### handle case in which the given or computed Hurst number falls outside
    #### the allowed range, 0 < H < 1, and correct those invalid values 
    #### in case they were computed; otherwise error message and terminate
    if acf =='ak' or acf =='AK':
        H = corr[2]  #Ver pk esta mal
        if H < 0.5:
            print('   ++ Hurst exponent H < 0.5 theoretically NOT allowed ++')
        if H >= 1:
            ###corr(3) = 0.99;
            print('   -- Hurstnumber computed/given: H > 1')
            print('      accepted, but outside the range of H [0;1]')
            ##print('      corrected to max. allowed H = 0.99('
        elif H <= 0: 
            corr[2] = 0.01
            print('   -- Hurstnumber computed: H < 0')
            print('      corrected to min. allowed H = 0.01')
            print('      NOTE: spectrum will be EXTREMELY flat')
            print('      generating a very heterogeneous slip model')
       
       
    #### print(' final spectral decay parameters to screen')
    if acf =='ex' or acf =='EX' or acf =='gs' or acf =='GS':
        print('   Final Decay Parameters:')
        print('       az = {0:.2f} km, ax = {1:.2f} km'.format(corr[0], corr[1]) )
    elif acf =='ak' or acf =='AK':
        print('   Final Decay Parameters:')
        print('       az = {0} km, ax = {1} km, H = {2:.2f}'.format(corr[0], corr[1], corr[2]))
#        print('       az = {0:.2f} km, ax = {1:.2f} km, H = {2:.2f}'.format(corr[0], corr[1], corr[2]))
    elif acf =='fr' or acf =='FR':
        print('   Final Decay Parameters:')
        print('       D = {0:.2f}, kc = {1:.2f}'.format(corr[0], corr[1]))
       
       
    #### ---------------------------------
    #### GENERATE THE SPATIAL RANDOM FIELD
    #### ---------------------------------
    #### call spectral synthesis function to create random field
    #### we use SpecSyn2 which allows to specify arbitrary sampling
    #### intervals in either direction
       
    if synnp2 =='y':
        print('    --->  use next power of two grid size for simulation')
        nx = L/samp[1]; nz = W/samp[0]
        nmax = max(nx, nz)
        ns = 2**nexp2(nmax)		## new grid-size
        xsc = ns/L;     zsc = ns/W
    
     #### now adjust correlation length to reflect the new size
        if acf =='ak':
            ncorr = [corr[0]*zsc, corr[1]*xsc, corr[2]]
        elif acf =='ex' or acf =='gs':
            ncorr = [corr[0]*zsc, corr[1]*xsc]
    
        #### simulate field, down-sample (interpolate) onto
        #### desired grid size, and set parameters correctly
        s,spar = SpecSyn2([ns, ns],[1, 1],ncorr,acf,SSseed)
        spar['synmeth'] ='synnp';spar['synN'] = ns
        spar['dim'] = [W, L]; 	 spar['samp'] = samp
        spar['lx'] = np.arange(0,((L-2)*samp[1]),samp[1])
#       spar['lx'] = np.arange(0,(L-1)*samp[1]+1,samp[1])
        spar['lz'] = np.arange(0,(W-1)*samp[0]+1,samp[0])
        spar['corr'] = corr;  	 spar['size'] = size(G) 
       
    else:
          
       ## this is the standard option to use
        G,spar = SpecSyn2([W, L],samp,corr,acf,SSseed)
       
    SSseed = spar['Rseed']		# save the seed value
    print(' This is the SEEEEEDDDD', SSseed)
    G = G - np.mean(G)		# to ensure that the simulated random field
    G = G/np.std(G,ddof=1)			# has zero mean and unit variance
       
       
    #### crude check that positive values are concentrated in the interior
    #### of the field; this should avoid large slip at the boundaries which
    #### will result in large stresses and unrealistically "patchy" slip models
    lz,lx = np.shape(G) 
    px = round(lx/np.sqrt(2))		# dimensions for('interior' area
    pz = round(lz/np.sqrt(2))	
    qx = np.floor(0.5*(lx-px))	# indices for('interior' area
    qz = np.floor(0.5*(lz-pz))		
    m,n= np.shape(G)
    GI = G[qz:m-qz,qx:n-qx]
    if np.mean(GI) < np.mean(G): G = -1*G
     
     
    #### ---------------------------------------------------------
    #### RESIZE AND SCALE FIELD TO MATCH DESIRED SOURCE PARAMETERS
    #### ---------------------------------------------------------
       
    #### resize grid (by bilinear interpolation) in case of subfault definition
    #### Note that spectral synthesis tool works on grid-dimensions of ODD size!
 
    print('UFF Este es grd', grd)

    if grd =='sub':
        print('   Slip defined on SUBFAULTS selected')
        G,spar['lx'],spar['lz'] = interpgrid2(G,spar['samp'],spar['dim'],spar['samp'])
    else:
        print('   Slip defined on GRID-NODES selected')
     
     
    #### perform a non-linear transformation, if desired, to create models
    #### with higher peak-slip values
    if nexp != 1:
        print('  Non-linear scaling of slip function: S=S**{0}'.format(nexp))
        G = G - np.min(G)
        G = G**nexp
        G = G - np.mean(G)
     
     
    #### calculating mean slip variation dsig and max printlacement dmax
    #### from empirical data; needs input D in cm;
    dsig,dmax,CSseed = CalcSigfromD(100*D,mech,CSseed)

    #### scale zero-mean, unit-variance random field by slip variation sigma 
    #### using twice the computed mean slip as allowed standard deviation
    #### D is assumed to be in meters, here we need D in cm!
    X = D*100*(1 + 2*G)		
     
     
    #### --------------------------------------------------------
    #### MAKE SLIP POSTIVE DEFINITE, AND APPLY TAPER TO THE EDGES
    #### --------------------------------------------------------
     
    #### two possibilities to make field positive-definite:
    #### 1) set all values < 0 (or any other cut-off value)
    ####    equal zero which effectively changes the spectral decay 
    #### 2) add min. value, i.e. "lift" field above zero level
    #### the first method yields patchier models with moment close to the
    #### desired one, the second preserves the heterogeneity spectrum 
#    if len(wlevel) == 0:
    if wlevel == -4:
        print('   Water-level: field simply(''lifted'' above zero')
        #### "lift" field above zero level
        Y = X - np.min(X)
        WLseed = []
    elif len(wlevel) != 0:
        if wlevel == 0:
            print('   Water-level: values < 0 set to zero')
            #### set values smaller 0 equal zero, 
            Y = X - np.min(X)
            [Y,WLseed] = WaterLevel(Y,'abs',abs(np.min(X)),'n',[])
        elif wlevel == -1:
        #### set values smaller 0 equal zero, with small random
        #### values of 0.05*max(field) added
            print('   Water-level: values < 0 set to small random value')
            Y = X - np.min(X)
            Y,WLseed = WaterLevel(Y,'abs',abs(np.min(X)),'y',WLseed)
        elif wlevel == -2:
        #### set values smaller 0 equal zero, with 0.25*field-value
        #### in place of the zeros
            print('   Water-level: values < 0 set to 0.25 their original value')
            Y = X - np.min(X)
            Y,WLseed = WaterLevel(Y,'abs',abs(np.min(X)),'d',[])
        elif wlevel == -3:
        #### use the computed mean slip (in cm) as cut-off
            print('   Water-level: values < D set to small random values')
            Y = X - np.min(X)
            Y,WLseed = WaterLevel(Y,'abs',100*D,'y',WLseed)
        else:
        #### use user-specified value as cut-off
            print('  Water-level: values < {0:d} set to small random values'.format(wlevel))
            Y = X - np.min(X)
            Y,WLseed = WaterLevel(Y,'abs',wlevel,'y',WLseed)
     
     
    #### taper slip at edges to obtain zero slip amplitudes at fault boundaries
    #### the taperwidth is specified in a vector of the form [left/right top bottom]
    #### Kaiser-window ('kw') yields very sharp edges, Hanning-window ('hn') is 
    #### less steep
    #### if the given taper vector has 4 elements, the last one (P) is used to 
    #### apply simple "depth-taper" by scaling the field as S**P in down-dip
    #### direction. P = 0.1-0.5 is a reasonable choice
    
    ##if W >= 0.95*depth
    ##  print('  ----------  SURFACE RUPTURE ---------- 
    ##  print('  rupture W >= 0.95*depth, taper adjusted 
    ##  print('  ---------------------------------------  
    ##  tapslp = [tapslp(1) 0 tapslp(3)];
    ##end

     
    if taper =='y' or taper =='d':
     
        print('   Slip function tapered at the boundaries:')
        print('** left/right, top, bottom: [ {0:.1f}, {1:.1f}, {2:.1f}] km'.format(twkm[0], twkm[1], twkm[2]))
        print('This is twkm[0]')
        print(twkm[0])
        print('y este es tapslp')
        T = TaperSlip(Y,tapslp,tapwin) 
     
        if 'twdpt' in locals():
            print(' Additional depth taper applied: Sz**{0}'.format(twdpt))
            i1,j1=np.shape(T)
            w1 = np.linspace(1,i1*samp[0],i1)
            w = np.transpose(w1**twdpt)
            w = w/max(w)
            for i in range(0,j1):
                T[:][:,i]=T[:][:,i]*w

        T = T - np.min(T)

        mo,mw = fmomentN(T,[W, L])
        Mo = fMw2MoN(Mw)
        S = T*Mo/mo
        d = np.mean(S)

    elif type(taper)==list and len(taper) >= 3:
        print('   Slip function tapered at the boundaries:')
        print('   left/right, top, bottom: [ {0:.1f}, {1:.1f}, {2:.1f}] km'.format(twkm[0], twkm[1], twkm[2]))
        T = TaperSlip(Y,tapslp,tapwin) 
     
        if 'twdpt' in locals():
            print(' Additional depth taper applied: Sz**{0}'.format(twdpt))
            i1,j1=np.shape(T)
            w1 = np.linspace(1,i1*samp[0],i1)
            w = np.transpose(w1**twdpt)
            w = w/max(w)
            for i in range(0,j1):
                T[:][:,i]=T[:][:,i]*w

        T = T - np.min(T)

        mo,mw = fmomentN(T,[W, L])
        Mo = fMw2MoN(Mw)
        S = T*Mo/mo
        d = np.mean(S)

    elif type(taper)==list and len(taper) == 0:
        print('   Slip function not tapered at the boundaries')
        T = Y

        T = T - np.min(T)
     
        mo,mw = fmomentN(T,[W, L])
        Mo = fMw2MoN(Mw)
        S = T*Mo/mo
        d = np.mean(S)

#    return S, par, per

    
    #### -------------------------------------
    #### CREATE FINAL OUTPUT ARRAYS/STRUCTURES
    #### -------------------------------------
   
    par={}
    par['txt0'] ='---------------------------------'
    par['txt1'] ='PARAMETERS FOR SIMULATED SLIP MAP'
    par['txt2'] ='Please store for later reference!'
    par['txt3'] ='---------------------------------'
    par['slip'] = S
    par['mech'] = mech
    par['dim'] = spar['dim']
    par['Mo'] = Mo
    par['Mw'] = Mw
    par['Dmean']= d
    par['dmaxmod'] = np.max(S)
    par['dmaxpred'] = dmax
    par['samp'] = spar['samp']
    par['corr'] = corr
    par['acf'] = acf
    par['depth']= depth
    par['dip'] = dip
    par['h'] = h
    par['lx'] = spar['lx']
    par['lz'] = spar['lz']
    par['grd'] = grd
    ff=spar.keys()
    if 'synmeth' in ff:
        par['synmeth'] = spar['synmeth']
    par['nexp'] = nexp
    par['wlevel'] = wlevel
    par['taper'] = twkm
    if 'twdpt' in locals():
        par['taper'] = [twkm, twdpt]
    par['Rseed.SS'] = SSseed 
    par['Rseed.WL'] = WLseed
    par['Rseed.CS'] = CSseed
    par['Rseed.RC'] = RCseed
    par['Rseed.RWC'] = RWCseed
    par['Rseed.CR'] = CRseed

    per={}
    per['Rseed.SS'] = SSseed
    per['Rseed.WL'] = WLseed
    per['Rseed.CS'] = CSseed
    per['Rseed.RC'] = RCseed
    per['Rseed.RWC'] = RWCseed
    per['Rseed.CR'] = CRseed
    f1=open('parALL1a','w')
    pprint(per,f1)
    f1.close()

    f2=open('parALL2','w')
    por1={'Rseed':{'SS': SSseed}}
    por2={'Rseed':{'WL': WLseed}}
    por3={'Rseed':{'CS': CSseed}}
    por4={'Rseed':{'RC': RCseed}}
    por5={'Rseed':{'RWC': RWCseed}}
    por6={'Rseed':{'CR': CRseed}}
    pprint(por1,f2)
    pprint(por2,f2)
    pprint(por3,f2)
    pprint(por4,f2)
    pprint(por5,f2)
    pprint(por6,f2)
    f2.close()

    pir={'Rseed':{'SS':SSseed,'WL':WLseed,'CS':CSseed,'RC':RCseed,'RWC':RWCseed,'CR':CRseed}}
    op2=open('md2aE.pkl','wb')
    pickle.dump(pir, op2)
    op2.close()

#    return S, par, per

    #### ------------------------
    #### DISK or GRAPHICAL OUTPUT
    #### ------------------------
    #### create output, if requested: write to disk
    #### or plot to screen
    
    #### write slip realization to output file if requested
    if outfile !='n':
        print('   Slip realization will be written to file:')
        print('  -->  {0}  <--'.format(outfile))
#        print ['  --> (',outfile,'  <--'])
        WriteArray(S,[W, L],samp,outfile)
   
    #### simple plot of resulting slip realization
    if fig =='y': 
        print('   Slip realization will be graphically displayed')

    return S, par, per

    lz,lx = np.shape(S)
    dz = W/lz
    dx = L/lx
    
#    %%% calculate Mw,Mo for slip distribution
    dim=[W, L]
    Mo,Mw = fmomentN(S,dim)
    
#    %%% set up axis
    zax = np.linspace(0,W,lz)
    xax = np.linspace(0,L,lx)
   

#********

    mmin=np.min(S)
    mmax=np.max(S)
    dm=mmax-mmin
#    print('mmin, mmax', mmin, mmax
    levels = np.arange(mmin,mmax, dm/9.0)

#    %% define contour intervals
    ci = np.floor(np.max(S)/10)
    if ci >= 10:
        ci = 10*np.round(ci/10)
    elif ci <=1:
        ci = 1



#    return S, par, per
