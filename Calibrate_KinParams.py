# Created by Tariq Anwar Aquib,2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np
import scipy
from scipy.stats import norm, lognorm, weibull_min
from scipy.stats import rv_continuous

import subprocess
import os

import skfmm


def rupvel2onsettime(rupvel, h, s):
    # rupvel is rupture velocity; it is a 2D array.
    # n is the size of rup_vel
    # h is grid spacing in meters.
    # s is the index of hypocentre; [ z, x]

    # extend input for 3d - raytracer supports only 3D input...
    n = rupvel.shape
    n = np.append(n, 1)
    s = np.append(s, 1)

    # Specify the subfolder where executable and binary files are located
    subfolder = './Raytracing'

    #convert and write rupvel to file in format for raytracer
    buffer = np.ndarray.flatten(rupvel,order='F') #Very important

    for j in range(0,n[1]):
        rec = (j)*n[0]

    bin_file_path = os.path.join('./Raytracing/rupvel.bin')
    bin_file_path1 = os.path.join('rupvel.bin')
    with open(bin_file_path, 'wb') as fid:
        fid.write(buffer.tobytes())

    #prepare input file for raytracer
    
    input_file_path = os.path.join('./Raytracing/input.inp')
    with open(input_file_path, 'w') as fid:
        fid.write(f'{bin_file_path1} \n')
        fid.write(f'{h} \n')
        fid.write(f'{n[0]} \n')
        fid.write(f'{n[1]} \n')
        fid.write(f'{n[2]} \n')
        fid.write('1 \n')  # only one 'source'
        fid.write(f'{s[0]} \n')
        fid.write(f'{s[1]} \n')
        fid.write(f'{s[2]} \n')

    # run the raytracer
    pathcwd = os.getcwd()
    os.chdir('./Raytracing/')  # Change directory
    
    executable_path = os.path.join('./raytracer.exe')
    ret = subprocess.call(executable_path)
    if ret != 0:
        print('Error running raytracer...')

    
    # read raytracer results and convert to output format
    out_file_path = os.path.join('./first_arrival.out')
    with open(out_file_path, 'rb') as fid:
        t = np.frombuffer(fid.read(), dtype='float32')

    print(os.getcwd())
    command = 'rm ' + 'first_arrival.out'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    os.chdir(pathcwd)
    T = np.reshape(t, (n[1], n[0]), order='C')  # Read along columns;Very important

    return T

def Calibrate_Vr(vr,Vr_mean,a_trunc=0,b_trunc=6):

    ### Define distribution names and corresponding functions
    distribution_names = ['normal', 'lognormal']
    #distribution_funcs = {'normal': norm, 'lognormal': lognorm}

    aic_values = np.zeros(len(distribution_names))

    # Fit distribution and calculate AIC and BIC
    pd = []

    for i, name in enumerate(distribution_names):
        if name == 'normal':
            params = norm.fit(vr.ravel())
            pd.append(norm(*params))
        elif name == 'lognormal':
            shape, loc, scale = lognorm.fit(vr.ravel(), floc=0)  # Lognormal fit
            pd.append(lognorm(shape, loc, scale))
    
        # Calculate negative log-likelihood
        neg_log_likelihood = -np.sum(pd[-1].logpdf(vr.ravel()))
    
        # Number of parameters
        num_params = 2 
    
        # Calculate AIC and BIC
        aic_values[i] = 2 * num_params - 2 * neg_log_likelihood

    # Select the distribution with the minimum AIC
    k = np.argmin(np.abs(aic_values))

    if k == 0: # Fit normal
        pd = []

        params = norm.fit(vr.ravel())
        pd.append(norm(*params))

        pd1 = pd[0]
        pd1.mean = Vr_mean

        loc = Vr_mean
        scale = params[1]

        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

        aa = scipy.stats.truncnorm.rvs(a,b,loc=loc,scale =scale,size=vr.size)
        # Here add an option of increasing threshold.!!

        vr_new = np.zeros_like(vr)

        vr_new[np.unravel_index(np.argsort(vr.flatten()), vr.shape)] = np.sort(aa)

    else: # Fit lognormal
        pd = []
        sigma, loc, scale = lognorm.fit(vr.ravel(), floc=0)  # Lognormal fit

        def trunc_lognorm_rvs(Vr_mean, sigma, loc, k_trunc, size=1):
            out = np.empty(0)
            while out.size != size:
                xx = scipy.stats.lognorm.rvs(loc=loc,scale=Vr_mean,s=sigma,size=size)
                out = np.append(out, xx[xx < k_trunc])[:size]
            return out

        aa = trunc_lognorm_rvs(Vr_mean, sigma, loc, b_trunc, size=vr.size)

        vr_new = np.zeros_like(vr)

        vr_new[np.unravel_index(np.argsort(vr.flatten()), vr.shape)] = np.sort(
                    aa)
        
    return vr_new


def Calibrate_Vmax_ratio(slip,psv,SP_ratio): # slip has to be slip from slip_ml (non-tapered)
    # First change PSV ratio; solves the issue of low psv values.!!
    psv_ratio = (slip/100)/psv  ### PSV is in m/sec while slip is cm/sec
    psv_ratio1 = psv_ratio
    psv_ratio = np.ndarray.flatten(psv_ratio)
    psv_ratio = psv_ratio[psv_ratio < SP_ratio]

    ### Define distribution names and corresponding functions
    distribution_names = ['normal', 'lognormal','weibull']

    aic_values = np.zeros(len(distribution_names))

    # Fit distribution and calculate AIC and BIC
    pd = []

    for i, name in enumerate(distribution_names):
        if name == 'normal':
            params = norm.fit(psv_ratio.ravel())
            pd.append(norm(*params))
        elif name == 'lognormal':
            params = lognorm.fit(psv_ratio.ravel(), floc=0)  # Lognormal fit
            pd.append(lognorm(*params))
        elif name == 'weibull':
            params = weibull_min.fit(psv_ratio.ravel(), floc=0)
            pd.append(weibull_min(*params))
            # Calculate negative log-likelihood

        neg_log_likelihood = -np.sum(pd[-1].logpdf(psv_ratio.ravel()))
        
        # Number of parameters
        num_params = 2 #len(params) if name == 'normal' else 3
        
        # Calculate AIC and BIC
        aic_values[i] = 2 * num_params - 2 * neg_log_likelihood

    # Select the distribution with the minimum AIC
    k = np.argmin(np.abs(aic_values))

    # if k==0: print('using normal')
    # elif k==1: print('using lognormal')
    # elif k==2: print('using weibull')

    c_trunc = SP_ratio
    size1 = psv_ratio1.size

    if k==0: # For Normal distribution
        pd = []
        params = norm.fit(psv_ratio.ravel())
        pd.append(norm(*params))

        pd1 = pd[0]
        loc = params[0]
        scale = params[1]

        a, b = (0 - loc) / scale, (c_trunc - loc) / scale

        aa = scipy.stats.truncnorm.rvs(a,b,loc=loc,scale =scale,size=size1)

        psv_ratio_new = np.zeros_like(psv_ratio1)

        psv_ratio_new[np.unravel_index(np.argsort(psv_ratio1.flatten()), psv_ratio1.shape)] = np.sort(aa)

    elif k==1:  # For lognormal distribution
        pd = []
        sigma, loc, scale = lognorm.fit(psv_ratio.ravel(), floc=0)  # Lognormal fit

        def trunc_lognorm_rvs(Vr_mean, sigma, loc, k_trunc, size=1):
            out = np.empty(0)
            while out.size != size:
                xx = scipy.stats.lognorm.rvs(loc=loc,scale=Vr_mean,s=sigma,size=size1)
                out = np.append(out, xx[xx < k_trunc])[:size]
            return out

        aa = trunc_lognorm_rvs(scale, sigma, loc, c_trunc, size=size1)

        psv_ratio_new = np.zeros_like(psv_ratio1)

        psv_ratio_new[np.unravel_index(np.argsort(psv_ratio1.flatten()), psv_ratio1.shape)] = np.sort(aa)   

    elif k==2: # For Weibull distriubtion
        params = weibull_min.fit(psv_ratio.ravel(), floc=0)
        aa = np.empty(0)
        
        while aa.size != size1:
            
            xx = scipy.stats.weibull_min.rvs(*params,size=size1)
            aa = np.append(aa, xx[xx < c_trunc])[:size1]

        psv_ratio_new = np.zeros_like(psv_ratio1)

        psv_ratio_new[np.unravel_index(np.argsort(psv_ratio1.flatten()), psv_ratio1.shape)] = np.sort(aa)

    return psv_ratio_new


def Calibrate_Vmax(psv1,a_psv,b_psv):
    distribution_names = ['normal', 'lognormal','weibull']

    aic_values = np.zeros(len(distribution_names))

    # Fit distribution and calculate AIC and BIC
    pd = []

    for i, name in enumerate(distribution_names):
        if name == 'normal':
            params = norm.fit(psv1.ravel())
            pd.append(norm(*params))
        elif name == 'lognormal':
            params = lognorm.fit(psv1.ravel(), floc=0)  # Lognormal fit
            pd.append(lognorm(*params))
        elif name == 'weibull':
            params = weibull_min.fit(psv1.ravel(), floc=0)
            pd.append(weibull_min(*params))
            # Calculate negative log-likelihood

        neg_log_likelihood = -np.sum(pd[-1].logpdf(psv1.ravel()))
        
        # Number of parameters
        num_params = 2 #len(params) if name == 'normal' else 3
        
        # Calculate AIC and BIC
        aic_values[i] = 2 * num_params - 2 * neg_log_likelihood

    # Select the distribution with the minimum AIC
    k = np.argmin(np.abs(aic_values))

    # if k==0: print('using normal')
    # elif k==1: print('using lognormal')
    # elif k==2: print('using weibull')

    size1 = psv1.size

    if k==0: # For Normal distribution
        pd = []
        params = norm.fit(psv1.ravel())
        pd.append(norm(*params))

        pd1 = pd[0]
        loc = params[0]
        scale = params[1]

        a, b = (a_psv - loc) / scale, (b_psv - loc) / scale

        aa = scipy.stats.truncnorm.rvs(a,b,loc=loc,scale =scale,size=size1)

        psv2 = np.zeros_like(psv1)

        psv2[np.unravel_index(np.argsort(psv1.flatten()), psv1.shape)] = np.sort(aa)

    elif k==1:  # For lognormal distribution
        pd = []
        sigma, loc, scale = lognorm.fit(psv1.ravel(), floc=0)  # Lognormal fit

        def trunc_lognorm_rvs(Vr_mean, sigma, loc, k_trunc, size=1):
            out = np.empty(0)
            while out.size != size:
                xx = scipy.stats.lognorm.rvs(loc=loc,scale=Vr_mean,s=sigma,size=size1)
                out = np.append(out, xx[xx < k_trunc])[:size]
            return out

        aa = trunc_lognorm_rvs(scale, sigma, loc, b_psv, size=size1)

        psv2 = np.zeros_like(psv1)

        psv2[np.unravel_index(np.argsort(psv1.flatten()), psv1.shape)] = np.sort(aa)   

    elif k==2: # For Weibull distriubtion
        params = weibull_min.fit(psv1.ravel(), floc=0)
        aa = np.empty(0)
        
        while aa.size != size1:
            
            xx = scipy.stats.weibull_min.rvs(*params,size=size1)
            aa = np.append(aa, xx[xx < b_psv])[:size1]

        psv2 = np.zeros_like(psv1)

        psv2[np.unravel_index(np.argsort(psv1.flatten()), psv1.shape)] = np.sort(aa) 

    return psv2

def compute_onset_times(vr,hyp_ind,dx):
    T = rupvel2onsettime(vr, dx, hyp_ind)
    T = np.transpose(T)

    # No longer using raytracing by Walter but rather python code online: https://github.com/scikit-fmm/scikit-fmm/tree/master

    return T

def compute_onset_times_scikit(vr,hyp_ind,dx):
    X, Y = np.meshgrid(np.linspace(-1,1,vr.shape[1]), np.linspace(-1,1,vr.shape[0]))
    phi = -1*np.ones_like(vr)
    phi[hyp_ind[0],hyp_ind[1]] = 1
    #d = skfmm.distance(phi, dx=float(dx))
    t = skfmm.travel_time(phi, vr, dx=float(dx))

    return t

def Compute_Tr(psv,slip,Tacc_ratio=0.1):
    Tr = ( (1.04*slip)/((Tacc_ratio)**(0.54)*psv) )**(1/1.01)
    return Tr

