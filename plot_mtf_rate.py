# Created by Tariq Anwar Aquib,2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
import sfft1
from sfft1 import sfft1
import matplotlib.pyplot as plt

def plot_mtf_rate(slip_time,slip_rate_yoffe,slip_rate_ml,samp):

    fac = 3.3e+10 * samp[0] * samp[1] * 1e6  # Convert to Nm
    mtf_yoffe = np.sum(np.sum(slip_rate_yoffe,axis=0),axis=0)*fac
    mtf_stf_ml = np.sum(np.sum(slip_rate_ml,axis=0),axis=0)*fac

    slip_time1 = np.linspace(start=slip_time[0],stop=60,num=1501)
    mtf_yoffe1 = np.zeros_like(slip_time1)
    mtf_stf_ml1 = np.zeros_like(slip_time1)

    mtf_yoffe1[:len(mtf_yoffe)] = mtf_yoffe
    mtf_stf_ml1[:len(mtf_stf_ml)] = mtf_stf_ml

    freq, amp_yoffe = sfft1(mtf_yoffe1)
    freq, amp_stf_ml = sfft1(mtf_stf_ml1)

    # plotting
    f, (ax1, ax2) = plt.subplots(1, 2, tight_layout=False,figsize=(5,3))

    ax1.plot(slip_time,mtf_yoffe, linewidth=2,label='Yoffe STF')
    ax1.plot(slip_time,mtf_stf_ml, linewidth=2,label='ML STF')
    ax1.set_xlim([0,15])
    #ax1.legend()
    ax1.set_ylabel('Moment rate Nm/sec')
    ax2.plot(freq, amp_yoffe,label='Yoffe STF')  
    ax2.plot(freq, amp_stf_ml,label='ML STF')      

    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    ax2.set_xlim([0.01,20])
    # ax2.set_ylim([0.1,10000])

    # Fit Brune's spectrum

    fc = np.linspace(start=0.01,stop=3,num=300)
    err_bfit = np.zeros_like(fc)
    for i in range(0,len(fc)):
        bv = 1/( (1 + (freq/fc[i])**2) )
        y = amp_yoffe/np.max(amp_yoffe) - bv
        err_bfit[i] = np.sqrt(np.mean(y**2))
    k = np.where(err_bfit==np.min(err_bfit))
    fc1 = fc[k]
    A = amp_yoffe[0]/((1 + (freq/fc1)**2 ))  
    #
    ax2.plot(freq,A,label='Brune model',linestyle='--')
    ax2.legend()
    f.suptitle(' Moment rate (Nm/sec) ', fontsize=12)