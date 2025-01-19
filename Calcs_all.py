# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np
def CalcDimfromM(*args):
#function [L,W,D,Mw,RCseed] = CalcDimfromM(Mw,mech,dscale,rel,RCseed);
#
# [L,W,D,Mw,RCseed] = CalcDimfromM(Mw,'mech','dscale',rel,RCseed) 
# computes the rupture dimensions given moment magnitude Mw and the 
# fault mechanism 'mech'. Data are based on scaling relations 
# established in Mai & Beroza (2000) or Wells & Coppersmith (1994).
# The source parameters are randomly picked within 95% confidence
# intervals of a normal distribution around the computed values.
#
#  input: Mw    - moment magnitude
#    mech   - 'ss' for strike-slip
#      'ds' for dip-slip
#      'al' for all events
#    dscale- 'd'  to scale mean slip such that target 
#           moment is reached exactly (default)
#      'm'  does not scale mean slip, and actual
#           moment may differ from target moment
#    rel  - 'MB' to use Mai & Beroza (2000) coefficients
#      'WC' to use Wells & Coppersmith (1994) coefficients
#
#    RCseed- optional 3x2-array for seed-values;
#      if given as 'off', the sample is given as computed,
#      without randomization (i.e. ignoring aleatory variability)
#
#  output: [L,W,D,Mw1] - fault length and width (in km)
#       mean slip (in m) resulting moment magnitude
#    RCseed-  3x2-array of seed-values for source realizations

#  Written by Martin Mai (mmai@pangea.Stanford.EDU) 
#  06/26/97
#  last change 08/28/2000
#  ------------------------------------------------

    size = len(args)
    
    if size == 2:
        Mw=args[0]; mech=args[1]; dscale = 'd'; rel = 'MB'; RCseed = 'off'
    elif size == 3:
        Mw=args[0]; mech=args[1]; dscale = args[2]; rel = 'MB'; RCseed = 'off'
    elif size == 4:
        Mw=args[0]; mech=args[1]; dscale = args[2]; rel = args[3]; RCseed = 'off'
    else:
        Mw=args[0]; mech=args[1]; dscale = args[2]; rel = args[3]; RCseed = args[4]
    
    mu = 3.3*1.0e10    # rigidity;
    source = ['eff', 'inv', 'inv']   
    # to use 'MB'-relations based on source dimension
    # NOTE that the "effective" dimensions may lead to 
    # small source size (hence large stress drop) while the
    # the dimensions from source inversions may overpredict 
    # the actual earthquake size; we therefore use a 
    # weighted average. {'eff' 'eff'} or {'inv' 'inv'} 
    # yield the dimenions for each case individually
    
    #%% define scaling constants, as published in Mai & Beroza (2000).
    if rel == MB:
                
        i = 0
        while i < len(source):
            #%% coefficients from dimensions used in source inversion
            if source[i] == 'inv':  
                if mech == 'al'   or mech == 'AL':
                    mL = 0.354;   bL = -5.204;  varL = 0.161
                    mW = 0.292;   bW = -4.281;  varW = 0.166
                    mA = 0.644;   bA = -9.444;  varA = 0.249
                    mD = 0.351;   bD = -4.982;  varD = 0.243
                elif mech == 'ss' or  mech == 'SS':
                    mL = 0.365;   bL = -5.149;   varL = 0.114
                    mW = 0.088;   bW = -0.540;   varW = 0.117
                    mA = 0.439;   bA = -5.594;   varA = 0.213  
                    mD = 0.549;   bD = -8.675;   varD = 0.199
                elif mech == 'ds' or  mech == 'DS':  
                    mL = 0.378;   bL = -5.713;   varL = 0.178
                    mW = 0.329;   bW = -4.927;   varW = 0.118
                    mA = 0.707;   bA = -10.640;  varA = 0.255  
                    mD = 0.294;   bD = -3.880;   varD = 0.255 
                     
      #%% coefficients from effective source dimensions
            elif source[i] == 'eff':
                if mech == 'al'   or mech == 'AL':
                    mL = 0.392;   bL = -6.127;   varL = 0.160 
                    mW = 0.323;   bW = -5.054;   varW = 0.161 
                    mA = 0.715;   bA = -11.181;  varA = 0.257 
                    mD = 0.285;   bD = -3.338;   varD = 0.257 
                elif mech == 'ss' or mech == 'SS':
                    mL = 0.404;   bL = -6.310;   varL = 0.118 
                    mW = 0.167;   bW = -2.183;   varW = 0.112 
                    mA = 0.571;   bA = -8.492;   varA = 0.189   
                    mD = 0.429;   bD = -6.026;   varD = 0.189 
                elif mech == 'ds' or mech == 'DS':
                    mL = 0.403;   bL = -6.388;   varL = 0.190 
                    mW = 0.350;   bW = -5.507;   varW = 0.148 
                    mA = 0.753;   bA = -11.896;  varA = 0.305   
                    mD = 0.247;   bD = -2.623;   varD = 0.305       
    
    #        end    #% condition on source-dimension type
                      
      #%% compute mean values of length, width, area, mean slip
            Lo[i] = mL*(1.5*Mw + 9.05) + bL 
            Wo[i] = mW*(1.5*Mw + 9.05) + bW 
            Ao[i] = mA*(1.5*Mw + 9.05) + bA 
            Do[i] = mD*(1.5*Mw + 9.05) + bD 
    
            i += 1
    #   end      #% loop over number in 'source'
    
        Lo = mean(Lo); Wo = mean(Wo) 
        Ao = mean(Ao); Do = mean(Do) 
    
    #%% Wells & Coppersmith coefficients; note that these coefficients 
    #%% are given w.r.t. to moment magnitude Mw, as in the publications,
    #%% in km (L, W) and m (D), respectively
    elif rel == 'WC':
        if mech == 'al'    or  mech == 'AL':
            mL = 0.59;   bL = -2.44;   varL = 0.16 
            mW = 0.32;   bW = -1.01;   varW = 0.15 
            mA = 0.91;   bA = -3.49;   varA = 0.24 
            mD = 0.69;   bD = -4.80;   varD = 0.36 
        elif  mech == 'ss' or mech == 'SS': 
            mL = 0.62;   bL = -2.57;   varL = 0.15 
            mW = 0.27;   bW = -0.76;   varW = 0.14 
            mA = 0.90;   bA = -3.42;   varA = 0.22  
            mD = 0.90;   bD = -6.32;   varD = 0.28 
        elif  mech == 'ds' or mech == 'DS':  
            mL = 0.58;   bL = -2.42;   varL = 0.16 
            mW = 0.41;   bW = -1.61;   varW = 0.15 
            mA = 0.98;   bA = -3.99;   varA = 0.26   
            mD = 0.08;   bD = -0.74;   varD = 0.38 
    
      #%% compute mean values of length, width, area, mean slip
        Lo = mL*Mw + bL 
        Wo = mW*Mw + bW 
        Ao = mA*Mw + bA 
        Do = mD*Mw + bD 
                        
    #%% call function NORMSAMPLE to find values for L, W, D out
    #%% of normal distributions; note that these values are the
    #%% logarithms of the actual source dimensions because the
    #%% scaling relations are defined based on log-log regression
    #if isempty(RCseed) == 1
    if RCseed == None:
        RCseed = np.zeros((3,2))
        dl,RCseed[0,] = NormSample(Lo,0.5*varL,100)
        dw,RCseed[1,] = NormSample(Wo,0.5*varW,100)
        da,RCseed[2,] = NormSample(Ao,0.5*varA,100)
        dd,RCseed[2,] = NormSample(Do,0.5*varD,100)
    else: 
        if type(RCseed) == str:
            if RCseed == 'off':
                dl = Lo
                dw = Wo
                da = Ao
                dd = Do
            else: print('No valid RCseed-value in CalcDimfromM')
        else:
            dl,RCseed[0,] = NormSample(Lo,0.5*varL,100,RCseed[0,])
            dw,RCseed[1,] = NormSample(Wo,0.5*varW,100,RCseed[1,])
            da,RCseed[2,] = NormSample(Ao,0.5*varA,100,RCseed[2,])
            dd,RCseed[2,] = NormSample(Do,0.5*varD,100,RCseed[2,])
    
        #%% compute "final" fault length, width and mean slip;
        #%% note that the results are more stable if A and W 
        #%% are estimated from scaling laws, and L is computed
        #% L = 10.^dl;
        W = 10**dw
        A = 10**da
        D = 10**dd
        if rel == 'WC':
            D = 100.0*D
        L = float(A)/W
    
    #%% check whether the values obtained for L,W, 
    #%% are compliant with fault area regression
    #% A = W*L;
    #% Aup = 10.^(Ao + varA); 
    #% Adown = 10.^(Ao - varA);
    #% if ((A > Aup) | (A < Adown))
    #%  disp('  CalcCorrfromM: Condition for fault area'); 
    #%  disp('  not fulfilled --> new sample is picked');  
    #%  dw = NormSample(Wo,0.5*varW,100);
    #%  W = 10.^dw;
    #%  A = L*W;
    #% end
    
    #%% calculate mean slip [in m] for chosen Mw and 
    #%% realization of fault dimensions 
    if dscale == 'd':
        #%% D scaled to meet the target Mw
        Mo = fMw2MoN(Mw)
        Mw = Mw
        D = (Mo/(mu*L*1e3*W*1e3))
    else:
        #%% D from distribution and 'random' Mw
        D = D/100
        Mo = mu*L*W*1e6*D
        Mw = fMo2MwN(Mo)

    return L,W,D,Mw,RCseed



import math as ma
import numpy as np
from NormSample import *
from mags import *
def CalcDimWC(*args):
#function [L,W,D,Mw,RWCseed] = CalcDimWC(L,W,mech,RWCseed);
#
#  function [L,W,D,Mw,RWCseed] = CalcDimWC(L,W,'mech',RWCseed) 
#  estimates the mean slip and seismic moment (moment
#  magnitude) for given fault length and width using the
#  empirical relation of Wells and Coppersmith.
#  The source parameters are randomly picked from a 
#  normal distribution around the computed values
#
#  input: L,W    - length and width, in km
#    mech    - 'ss' or 'SS' for strike-slip
#       'ds' or 'DS' for dip-slip
#       'al' or 'AL' for all events
#    RWCseed- optional 3x2-array for seed-values
#
#  output: [L,W,D,Mw] - fault length and width (in km)
#      mean slip (in m),resulting moment magnitude
#    RWCseed-  3x2-array of seed-values for source realizations

#  Written by Martin Mai (mmai@pangea.Stanford.EDU) 
#  09/21/99
#  ------------------------------------------------

    size = len(args)
    if size == 3:
        L=args[0]; W=args[1]; mech= args[2]; RWCseed=None
    elif size == 4:
        L=args[0]; W=args[1]; mech= args[2]; RWCseed= args[3]
    
    mu1 = 3.3*1e10    # rigidity;
    
    #%% define scaling constants, as published in Wells and Coppersmith (1994)
    
    if mech == 'ss'  or mech == 'SS': 
        mL = 1.49;   bL = 4.33;   varL = 0.24
        mW = 2.59;   bW = 3.80;   varW = 0.45
        mA = 1.02;   bA = 3.98;   varA = 0.23
    elif mech == 'ds'  or mech == 'DS':
        mL = 1.49;   bL = 4.49;   varL = 0.26
        mW = 1.95;   bW = 4.37;   varW = 0.32
        mA = 0.90;   bA = 4.33;   varA = 0.25  
    elif mech == 'al' or mech == 'AL':
        mL = 1.49;   bL = 4.38;   varL = 0.26
        mW = 2.25;   bW = 4.06;   varW = 0.41
        mA = 0.98;   bA = 4.07;   varA = 0.24  
    
    #%% compute "central" values of length, width, area, mean slip
    Mlo = bL + mL*ma.log10(L)
    Mwo = bW + mW*ma.log10(W)
    Mao = bA + mA*ma.log10(L*W)
    
    
    #%% call function NORMSAMPLE to find values for the moments estimated
    #%% from the given fault length, and area from normal distributions; 
    if RWCseed == None:
        RWCseed = np.zeros((3,2))
        Ml,RWCseed[0,] = NormSample(Mlo,0.5*varL,100)
        Mw,RWCseed[1,] = NormSample(Mwo,0.5*varW,100)
        Ma,RWCseed[2,] = NormSample(Mao,0.5*varA,100)
    else:
        Ml,RWCseed[0,] = NormSample(Mlo,0.5*varL,100,RWCseed[0,])
        Mw,RWCseed[1,] = NormSample(Mwo,0.5*varW,100,RWCseed[1,])
        Ma,RWCseed[2,] = NormSample(Mao,0.5*varA,100,RWCseed[2,])
    
    
    #%% use median value of the three estimates
    Mw = np.median([Ml, Mw, Ma])
    
    
    #%% calculate mean slip [in m] for chosen Mw and 
    #%% realization of fault dimensions 
    Mo = fMw2MoN(Mw)
    D = (Mo/(mu1*L*1e3*W*1e3))

    return L,W,D,Mw,RWCseed

import numpy as np
def CalcCorrfromLW(*args):
#function [Corr,CRseed] = CalcCorrfromLW(srcpar,mech,acf,CRseed);
#
#  function [Corr,CRseed] = CalcCorrfromLW(srcpar,'mech','acf',CRseed)
#  calculates the correlation len or fractal dimension
#  (i.e. spectral decay) of the power spectrum ACF for
#  a given moment. The final value is picked from a normal 
#  distribution within the error bounds of measured correlation
#  len and fractal dimensions.
#
#  input: srcpar - [W L M] for fault len, width, magnitude
#    mech   - 'ss' for strike-slip events; 
#      'ds' for dip-slip events
#    acf   - string to denote autocorrelation function
#      'gs' for Gaussian
#      'ex' for exponential
#      'ak' for anisotropic von Karman
#      'fr' for the fractal case
#    CRseed- optional array of seed-values;
#      size 1x2 for 'fr'
#      size 4x2 for 'gs','ex',
#        size 5x2 for 'ak'
#      if given as 'off', the sample is given as computed,
#      without randomization (ignoring aleatory variability)
#
#  output: Corr  - vector of correlation len Cz, Cx
#      for anisotropic von Karman: Corr = [Cz Cx H]
#            for Gauss or exponetial: Corr = [Cz Cx]
#      for fractal case: Corr = D;

#  Written by Martin Mai (mmai@pangea.Stanford.EDU) 
#  07/06/98
#  last change 03/01/2000 - saving seed-values added
#  ------------------------------------------------

    if len(args) <= 3:
        srcpar=args[0] 
        mech=args[1]
        acf=args[2]
        CRseed='off'
    else:
        srcpar=args[0] 
        mech=args[1]
        acf=args[2]
        CRseed=args[3]
    #%% check source parameter input
    if len(srcpar) == 1:
        print('*** CalcCorrfromLW stopped: not enough srcpar-input values given *** ')
    elif len(srcpar) == 2:
        W = srcpar[0]
        L = srcpar[1]
        print('**** magnitude computed from source-scaling relations***')
        L,W,D,M,RWCseed = CalcDimDMWC(L,W,mech,'off')
        print('M = {0:.2f}'.format(M))
    else:
        W = srcpar[0]
        L = srcpar[1]
        M = srcpar[2]
    
    
    #%% constants calculated from lsqr-fits of measured correlation len
    
    if acf == 'gs' or acf == 'GS':
        print('++ Coefficients for corr. lens of Gaussian ACF ++')
        print('++ are fixed to increase slowly with EQ-magnitude ++')
       
        bxM = -1.500;   mxM = 0.300;   varxM = 0.1
        bxL =  3.000;   mxL = 0.015;   varxL = 0.4
        bxW =  3.300;   mxW = 0.015;   varxW = 0.4
        bzM = -1.500;   mzM = 0.300;   varzM = 0.1
        bzL =  2.900;   mzL = 0.010;   varzL = 0.4
        bzW =  3.000;   mzW = 0.015;   varzW = 0.4
    
    
    elif acf == 'ex' or acf == 'EX':
    
        if mech == 'ss' or mech == 'SS':
            bxM = -3.163;   mxM = 0.625;   varxM = 0.14 
            bxL =  0.737;   mxL = 0.392;   varxL = 2.24      
            bxW = -4.949;   mxW = 1.815;   varxW = 6.01   
            bzM = -1.600;   mzM = 0.330;   varzM = 0.12 
            bzL =  2.422;   mzL = 0.063;   varzL = 1.08   
            bzW = -0.035;   mzW = 0.434;   varzW = 0.27 
         
        elif mech == 'ds'  or  mech == 'DS': 
            bxM = -2.571;   mxM = 0.522;   varxM = 0.15 
            bxL = -0.210;   mxL = 0.430;   varxL = 1.68         
            bxW =  0.499;   mxW = 0.629;   varxW = 4.51 
            bzM = -1.851;   mzM = 0.394;   varzM = 0.15   
            bzL =  1.211;   mzL = 0.241;   varzL = 3.16   
            bzW =  0.646;   mzW = 0.397;   varzW = 0.72 
    
        elif mech == 'al'  or mech == 'AL': 
            bxM = -2.811;   mxM = 0.565;   varxM = 0.15 
            bxL =  0.079;   mxL = 0.415;   varxL = 2.00   
            bxW =  4.865;   mxW = 0.591;   varxW = 7.34 
            bzM = -1.832;   mzM = 0.378;   varzM = 0.18   
            bzL =  1.095;   mzL = 0.168;   varzL = 4.16       
            bzW =  0.427;   mzW = 0.403;   varzW = 0.56 
          
    elif acf == 'ak' or acf == 'AK':
      
        meanH = 0.77; varH = 0.23
    
        if mech == 'ss' or mech == 'SS':
            bxM = -2.928;   mxM = 0.588;   varxM = 0.19 
            bxL =  1.855;   mxL = 0.341;   varxL = 2.79       
            bxW = -4.870;   mxW = 1.741;   varxW = 4.61 
            bzM = -1.619;   mzM = 0.328;   varzM = 0.15 
            bzL =  2.247;   mzL = 0.059;   varzL = 1.41   
            bzW = -0.386;   mzW = 0.438;   varzW = 0.75 
    
        elif mech == 'ds' or mech == 'DS':
            bxM = -2.433;   mxM = 0.492;   varxM = 0.14 
            bxL =  1.096;   mxL = 0.314;   varxL = 1.26         
            bxW =  1.832;   mxW = 0.449;   varxW = 3.87   
            bzM = -1.785;   mzM = 0.377;   varzM = 0.17 
            bzL =  1.118;   mzL = 0.208;   varzL = 2.87   
            bzW =  0.584;   mzW = 0.347;   varzW = 0.76 
    
        elif mech == 'al' or mech == 'AL':
            bxM = -2.595;   mxM = 0.527;   varxM = 0.19 
            bxL =  1.541;   mxL = 0.326;   varxL = 2.31   
            bxW =  6.072;   mxW = 0.416;   varxW = 6.80 
            bzM = -1.801;   mzM = 0.367;   varzM = 0.18
            bzL =  1.074;   mzL = 0.147;   varzL = 3.61
            bzW =  0.552;   mzW = 0.349;   varzW = 0.78
    
        elif acf == 'fr' or acf == 'FR':
            meanD = 2.29; varD = 0.23
    
    #%% obtain fractal dimensions and/or correlation len 
    #%% by successive call to function NORMSAMPLE
    
    #%% compute fractal dimension in case of fractal slip
    if acf == 'fr' or acf == 'FR':
        kc = 2*np.pi/(sqrt(L*W))  # corner wavenumber
#        if CRseed == None:
        if len(CRseed) == 0:
            CRseed = np.zeros(2)
            D,CRseed[0,] = NormSample(meanD,0.5*varD,100)
             
#        elif CRseed != None: 
        if len(CRseed) != 0:
            if type(CRseed) == str:
                if CRseed == 'off':
                    D = meanD
                else: print('**no valid CRseed-value in CalcCorrfromLW**')
            else:
                D,CRseed[0,] = NormSample(meanD,0.5*varD,100,CRseed)
    
    #%% compute correlation lengths
         
    else:
        CxM = bxM + mxM*M
        CxL = bxL + mxL*L
        CxW = bxW + mxW*W
        CzM = bzM + mzM*M
        CzL = bzL + mzL*L  
        CzW = bzW + mzL*W
    
        if len(CRseed) == 0:
            if acf == 'ak' or acf == 'AK':
                CRseed = np.zeros((7,2))
            else:
                CRseed = np.zeros((6,2))
    
            #%% along-strike direction, from len and width
            dxM,CRseed[0,] = NormSample(CxM,0.5*varxM,100)
            dxL,CRseed[1,] = NormSample(CxL,0.5*varxL,100)
            dxW,CRseed[2,] = NormSample(CxW,0.5*varxW,100)
      
            #%% down-dip direction, from len and width
            dzM,CRseed[3,] = NormSample(CzM,0.5*varzM,100)
            dzL,CRseed[4,] = NormSample(CzL,0.5*varzL,100)
            dzW,CRseed[5,] = NormSample(CzW,0.5*varzW,100)
         
        elif len(CRseed) != 0:
            if type(CRseed) == str:
                if CRseed == 'off':
                    dxL = CxL;  dxW = CxW;  dxM = CxM
                    dzL = CzL;  dzW = CzW;  dzM = CzM
                else: print(' **no valid CRseed-value in CalcCorrfromLW**') 
                 
            else:
    
                #%% along-strike direction, from len and width
                dxM,CRseed[0,] = NormSample(CxM,0.5*varxM,100,CRseed[0,])
                dxL,CRseed[1,] = NormSample(CxL,0.5*varxL,100,CRseed[1,])
                dxW,CRseed[2,] = NormSample(CxW,0.5*varxW,100,CRseed[2,])
    
                #%% down-dip direction, from len and width
                dzM,CRseed[3,]= NormSample(CzM,0.5*varzM,100,CRseed[3,])
                dzL,CRseed[4,]= NormSample(CzL,0.5*varzL,100,CRseed[4,])
                dzW,CRseed[5,]= NormSample(CzW,0.5*varzW,100,CRseed[5,])
      
        #%% use mean values for the correlation len, as obtained
        #%% from parametrization on fault len and fault width
        dxM = 10**dxM; dzM = 10**dzM
        Cx = np.mean([dxM, dxL, dxW])
        Cz = np.mean([dzM, dzL, dzW])
    
        #%% compute Hurstnumber H for anisotropic von Karman
        if acf == 'ak' or acf == 'AK':
            if type(CRseed) == str:
                if CRseed == 'off':
                    H = meanH
                else: print('ERROR ** No valid CRseed-value in CalcCorrfromLW')
                return
            else:
                if CRseed[6,] == np.zeros(2):
#                if CRseed[6,] == [0,0]:
#                if CRseed[6,] == np.zeros((1,2)):
                    H,CRseed[6,] = NormSample(meanH,0.5*varH,100)
                else:
                    H,CRseed[6,] = NormSample(meanH,0.5*varH,100,CRseed[6,])
    
    #%% asemble final output vecor
    if acf == 'gs' or acf == 'ex' or acf == 'GS' or acf == 'EX':
        Corr = [Cz, Cx]
    elif acf == 'ak' or acf == 'AK':
        Corr = [Cz, Cx, H]
    elif acf == 'fr' or acf == 'FR':
        Corr = [D, kc]

    return Corr,CRseed



def CalcSigfromD(*args):
#function [sigma,dmax,CSseed] = CalcSigfromD(D,mech,CSseed);
#
#  function [sigma,dmax,Rseed] = CalcSigfromD(D,'mech',CSseed) 
#  calculates the slip variation around the mean. The scaling of
#  displacement variability depends on moment and/or average 
#  displacement where the latter provides higher correlation, 
#  and hence a better way for prediction.
#  The actual value is picked from a normal distribution 
#  within the error bounds of measured sig. vs. Mo.
#  The same method is applied to compute a max. value for
#  the displacement.
#
#  See also NORMSAMPLE
#
#  input: D      - mean slip (in cm)
#    mech    - 'ss' or 'SS' for strike-slip events; 
#             'ds' or 'DS' for dip-slip events
#             'al' or 'AL' for all events
#    CSseed - optional: 2x2-array of seed values for random
#       number generator
#       if given as 'off', the sample is given as computed,
#       without randomization (ignoring aleatory variability)
#
#  output: sigma  - slip variation on fault plane (in cm)
#     dmax   - max. displacement on fault plane (in cm)
#     CSseed - 2x2-array of seeds for the random number generator

#  Written by Martin Mai (mmai@pangea.Stanford.EDU) 
#  06/26/97
#  last change 11/08/99
#  ------------------------------------------------

    if len(args) == 2:
        D=args[0]; mech=args[1]; CSseed = [] 
    else:
        D=args[0]; mech=args[1]; CSseed = args[2]
    #%% constants derived from regression analysis
    
    #%% for slip variation
    if mech == 'al' or mech == 'AL':
        ads = 0.809;    bds = 0.2942;  vards = 0.14
    elif mech == 'ss' or mech == 'SS':
        ads = 0.687;    bds = 0.516;   vards = 0.14
    elif mech == 'ds' or mech == 'DS':
        ads = 0.969;    bds = -0.123;  vards = 0.12
                              
    #%% for max. displacement
    if mech == 'al' or mech == 'AL':
        mds = 0.783;    xds = 1.024;  vrds = 0.16
    elif mech == 'ss' or mech == 'SS':
        mds = 0.677;    xds = 1.2078; vrds = 0.16
    elif mech == 'ds' or mech == 'DS':
        mds = 0.969;    xds = 0.684;  vrds = 0.15
    
    #%% compute central values for variation and max. displacement
    logsig = ads*math.log10(D) + bds
    logmax = mds*math.log10(D) + xds
    
    #%% pick sigma value from normal distribution
    if len(CSseed) == 0:
        CSseed = np.zeros((2,2))
        sigma,CSseed[0,] = NormSample(logsig,0.5*vards,100)
        dmax,CSseed[1,]  = NormSample(logmax,0.5*vrds,100)
               
    elif len(CSseed) != 0:
        if type(CSseed) == str:
            if CSseed == 'off':
                sigma=logsig
                dmax = logmax
            else: print('*** ERROR *** No valid CSseed-value in CalcSigfromD ****')
            return
                
        else:
            sigma,CSseed[0,] = NormSample(logsig,0.5*vards,100,CSseed[0,])
            dmax,CSseed[1,]  = NormSample(logmax,0.5*vrds,100,CSseed[1,])
                
        sigma = 10**sigma
        dmax = 10**dmax

    return sigma,dmax,CSseed


from NormSample import *
def CalcSigfromM(*args):
#function [sigma] = CalcSigfromM(D,mech);
#
#  function sigma = CalcSigM(D,'mech') calculates
#  the slip variation around the mean given by the
#  rupture dimensions and the moment of the rupturing fault
#  The actual value is picked from a normal distribution 
#  within the error bounds of measured sig. vs. Mo
#
#  input:  D     - mean slip (in meters!)
#          mech   - 'ss' or 'SS' for strike-slip events; 
#        'ds' or 'DS' for dip-slip events
#        'al' or 'AL' for all events
#

#  Written by Martin Mai (mmai@pangea.Stanford.EDU) 
#  06/26/97
#  last change 08/23/99
#  ------------------------------------------------

    size = len(args)
    if size == 2:
        D=args[0]; mech=args[1]

#%% constants derived from regression analysis
    if mech == 'al' or mech == 'AL':
        ads = 0.7028;  bds = 9.8271;  vards = 19.4544
    elif mech == 'ss' or mech == 'SS': 
        ads = 0.6759;  bds = 10.455;  vards = 16.9078
    elif mech == 'ds' or mech == 'DS': 
        ads = 0.7378;  bds = 8.2825;  vards = 22.8696
    
    #%% compute central value
    sigmaS = ads*D + bds
    
    #%% pick sigma value from normal distribution
    sigma = NormSample(sigmaS,0.5*vards,100)

    return sigma
