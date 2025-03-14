# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa


# Get first iteration of Vr and Vmax.
import numpy as np
from importlib import reload

import models_Vr_PSV
reload(models_Vr_PSV)
from models_Vr_PSV import *


import scipy.fftpack
import scipy.io


def Slip2Stress(slip,rake,samp):
    '''
    This code is a python implementation of Martin's Slip2Stress matlab code.
    Takes input of slip in cm/sec; rake in radians and sampling size in km, eg: [0.1,0.1]

    Author - Tariq Anwar Aquib
    Contact - tariqanwar.aquib@kaust.edu.sa
    Date: 09/01/2024
    '''

    # have to increase the size of slip to avoid discontinuity at borders..
    # Replicate rows in slip.
    slip_s1 = np.vstack([np.tile(slip[0, :], (10, 1)), slip, np.tile(slip[-1, :], (10, 1))])

    # Replicate columns in slip.
    slip_s2 = np.hstack([np.transpose(np.tile(slip_s1[:, 0], (10, 1))), slip_s1, np.transpose(np.tile(slip_s1[:, -1], (10, 1)))])

    # Computing strike and dip components of slip.
    SS = slip_s2*np.cos(rake*3.1415/180)
    DS = slip_s2*np.sin(rake*3.1415/180)

    # Computation of stress drop
    # input parameters for slip2stress code
    slipS = SS
    slipD = DS

    rig = np.array([3.3e10])
    lam = rig
    sfac = np.array([0.5])

    dim = np.zeros([2])
    dim[0] = ((slipS.shape[0]) -1)*samp[0]
    dim[1] = ((slipS.shape[1]) -1)*samp[1]


    # switch to SI units
    slipS = 0.01*slipS
    slipD = 0.01*slipD
    samp = 1000*samp

    # Determine FFT grid size
    N = 2 ** int(np.ceil(np.log2(max(slipS.shape[0], slipS.shape[1]))))


    # Get the 2D Fourier amplitude spectra..
    FT2DS = samp[0]*samp[1]* np.fft.fft2(slipS,s=(N,N))
    FT2DD = samp[0]*samp[1]* np.fft.fft2(slipD,s=(N,N))

    PHS = np.angle(FT2DS)
    PHD = np.angle(FT2DD)

    AMS = np.fft.fftshift(np.abs(FT2DS))
    AMD = np.fft.fftshift(np.abs(FT2DD))


    # Determine wavenumbers now
    knyz = 1/(2*samp[0])
    knyx = 1/(2*samp[1])

    nfz = AMS.shape[0]
    nfx = AMS.shape[-1]

    if np.mod(nfx,2) == 0:
        fx = np.linspace(-knyx,knyx,num=nfx+1)
        fx = fx[:-1]
    else:
        fx = np.linspace(-knyx,knyx,num=nfx)

    if np.mod(nfz,2) == 0:
        fz = np.linspace(-knyz,knyz,num=nfx+1)
        fz = fz[:-1]
    else:
        fz = np.linspace(-knyz,knyz,num=nfx)

    # set-up meshgrid to calculate stress drop
        
    [kx,kz] = np.meshgrid(fx,fz)
    k = np.sqrt(kx**2 + kz**2)

    # Find indices of zero wavenumber and exchange k there with a dummy value to avoid division by zero.
    # (K is set to zero at this point later anyway)
    eps = 2.2204e-16
    [zz,zx] = np.where(k==0)
    k[zz,zx] = eps

    # Static stiffness function
    # As originally used by Martin, assuming lam=rig:
    #   KS = (sfac*rig./k).*((4/3).*kx.^2 + kz.^2);
    #   KD = (sfac*rig./k).*(kx.^2 + (4/3).*kz.^2);
    # As in Andrews, 1980 JGR, eq. (20):

    KS = (sfac*rig/k)*( (2*(lam+rig)/(lam+2*rig)) *kx**2 + kz**2 )
    KD = (sfac*rig/k)*( (2*(lam+rig)/(lam+2*rig)) *kz**2 + kx**2 )
    KSD = (sfac*rig/k)*( (2*(lam+rig)/(lam+2*rig))-1 )*kx*kz

    # set K to zero for kx=kz=k=0
    KS[zz,zx] = 0
    KD[zz,zx] = 0
    KSD[zz,zx] = 0

    # static stress drop on the fault in FT-domain (Andrews 1980 eq. (19))

    SDFS = AMS*KS
    SDFD = AMD*KD
    SDFSD = AMS*KSD
    SDFDS = AMD*KSD

    # Static stress drop on fault in space domain (in Pa)
    i = 0+1j

    FS = np.fft.ifftshift(SDFS)*np.exp(i*PHS)
    FD = np.fft.ifftshift(SDFD)*np.exp(i*PHD)
    FSD = np.fft.ifftshift(SDFSD)*np.exp(i*PHS)
    FDS = np.fft.ifftshift(SDFDS)*np.exp(i*PHD)

    sigmaS = np.real(np.fft.ifft2(FS)) + np.real(np.fft.ifft2(FDS))
    sigmaD = np.real(np.fft.ifft2(FD)) + np.real(np.fft.ifft2(FSD))

    # Cut to original size
    sigmaS = sigmaS[:slipS.shape[0],:slipS.shape[1]]
    sigmaD = sigmaD[:slipS.shape[0],:slipS.shape[1]]
    # Correction factor 
    sigmaS = 2*np.pi*sigmaS
    sigmaD = 2*np.pi*sigmaD

    # Account for spatial sampling
    sigmaS = sigmaS/(samp[0]*samp[1])
    sigmaD = sigmaD/(samp[0]*samp[1])

    # From pa to MPa
    sigmaS = sigmaS/1e6
    sigmaD = sigmaD/1e6
    
    sigmaS = sigmaS[10:-10,10:-10]
    sigmaD = sigmaD[10:-10,10:-10]

    return sigmaS, sigmaD


def compute_CL(slip,hyp,dim,samp):

    nz,nx = slip.shape

    L = dim[1]
    W = dim[0]

    
    x = np.zeros(slip.shape[1])

    for i in range(hyp[1]+1,len(x)):
        x[i] = (i-hyp[1])*samp[1]

    for i in range(hyp[1]):
        x[i] = (i-hyp[1])*samp[1]


    y = np.zeros(slip.shape[0])

    for i in range(hyp[0]+1,len(y)):
        y[i] = -((i-hyp[0])*samp[0])

    for i in range(hyp[0]):
        y[i] = (hyp[0]-i)*samp[0]



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


def get_Vr_PSV(slip,hypo_x,hypo_z,lx,lz,samp,rake):

    nz,nx = slip.shape

    hyp_ind_x = np.argmin(np.abs(lx-hypo_x))
    hyp_ind_z = np.argmin(np.abs(lz-hypo_z))

    # compute stress drop
    sigmaS0, sigmaD0 = Slip2Stress(slip,rake,samp)

    sigmaS0 = sigmaS0/np.max(np.abs(sigmaS0))
    sigmaD0 = sigmaD0/np.max(np.abs(sigmaD0))

    # compute X, Y and crack lengths..

    hyp_ind = np.array([hyp_ind_z, hyp_ind_x])  

    X20, Y20, crack_length0 = compute_CL(slip,hyp_ind,[lz[-1],lx[-1]],samp)
    
    L = lx[-1]
    crack_length0 = crack_length0/L
    # Rupture velocity
    path_model = './Trained_models/low_minmax_unfiltered_sd'

    sigmaS = torch.from_numpy(sigmaS0); sigmaD = torch.from_numpy(sigmaD0); X2 = torch.from_numpy(X20); Y2 = torch.from_numpy(Y20); crack_length = torch.from_numpy(crack_length0)

    test_inputs = torch.cat([ sigmaS.unsqueeze(-1),  sigmaD.unsqueeze(-1), X2.unsqueeze(-1), Y2.unsqueeze(-1), crack_length.unsqueeze(-1) ],dim=-1)

    vr_org = get_vr_estimations(path_model,test_inputs,modes=8,width=16).cpu()


    # Peak slip velocity
    path_model = './Trained_models/Vr_PSV_4inputs_minmax'

    test_inputs = torch.cat([vr_org.unsqueeze(-1), X2.unsqueeze(-1),  Y2.unsqueeze(-1),  crack_length.unsqueeze(-1)],dim=-1)

    pred_gau = get_psv_estimations(path_model,test_inputs,modes=15,width=16)

    psv_fno = torch.abs(pred_gau.squeeze(0).squeeze(0).cpu())*torch.max(torch.from_numpy(slip/100))

    return vr_org, psv_fno, X20, Y20, crack_length0
