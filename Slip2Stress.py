# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
import scipy.fftpack
import scipy.io
#import matplotlib.pyplot as plt

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
