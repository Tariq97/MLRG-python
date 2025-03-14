# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Modified by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np

import distributions_hyp as pd

def Stat_Hypo(slip,lx,lz,mech):

    dim = slip.shape
    dim_z = dim[0]; dim_x = dim[1]

    ### CALCULATES PDF BASED ON IN-PLANE POSITION

    if mech == 'dp':
        hypz=pd.gampdf(np.linspace(0,1,dim_z),7.364,0.072)
    elif mech == 'dc':
        hypz=pd.wblpdf(np.linspace(0,1,dim_z),0.692,3.394)
    elif mech == 'ds':
        hypz=pd.gampdf(np.linspace(0,1,dim_z),12.658,0.034)
    elif mech == 'ss':
        hypz=pd.wblpdf(np.linspace(0,1,dim_z),0.626,3.921)
    elif mech == 'al':
        hypz=pd.wblpdf(np.linspace(0,1,dim_z),0.612,3.353)

    # --> along dip
    hypz=hypz.reshape(len(hypz),1)
        
    # --> along strike
    hypx=pd.normpdf(np.linspace(0,1,dim_x),0.5,0.23)
    hypx=hypx.reshape(1,len(hypx))

    hyp_pos = np.matmul(hypz,hypx)

    hyp_pos = hyp_pos/np.max(hyp_pos)

    #### CALCULATES PDF BASED ON MAXIMUM and MEAN SLIP RATIO
    # compute mean and max slip where slip is not zero
    mean_slip = np.mean(np.positive(slip))
    max_slip = np.max(np.positive(slip))

    # compute pdf for mean slip ratio
    tmp_slip=slip/mean_slip
    mm,nn=np.shape(tmp_slip)
    mech='ss'
    if  mech == 'dp' or mech == 'dc' or mech == 'ds':
        hyp_mean=pd.gampdf2(tmp_slip,2.616,0.623)
    elif mech == 'ss':
        hyp_mean=pd.gampdf2(tmp_slip,1.928,0.868)
    elif mech == 'al':
        hyp_mean=pd.gampdf2(tmp_slip,2.210,0.748)
        
    hyp_mean=hyp_mean.reshape((mm,nn))

    # compute pdf for max slip ratio
    tmp_slip=slip/max_slip
    mm,nn=np.shape(tmp_slip)
    if mech == 'dp' or mech == 'dc' or mech == 'ds':
        hyp_max=pd.wwbl2(tmp_slip,0.454,1.874)
    elif  mech == 'ss':
        hyp_max=pd.wwbl2(tmp_slip,0.446,1.551)
    elif  mech == 'al':
        hyp_max=pd.wwbl2(tmp_slip,0.450,1.688)
        
    hyp_max=hyp_max.reshape((mm,nn))

    # compute probability based on both previous slip ratios and scales to 1
    hyp_slip = hyp_max*hyp_mean
    hyp_slip = hyp_slip/np.max(hyp_slip)

    # computes probability based on both on-plane position and slip ratios
    slip_pos = hyp_slip*hyp_pos
    slip_pos = slip_pos/np.max(slip_pos)

    # define asperities regions (definitions from Mai et al. 2005)
    out_asp_mask = slip < 0.33 * max_slip
    asp_mask = (slip >= 0.33 * max_slip) & (slip < 0.66 * max_slip)
    big_asp_mask = slip >= 0.66 * max_slip;

    # Define asperities positions on the fault plane (0=no asperity, 1=asperity, 2=big asperity)
    asp_fault = np.zeros_like(slip, dtype=int); 
    asp_fault[asp_mask] = 1;        
    asp_fault[big_asp_mask] = 2;   

    # refreshes hypocenter locations probability according to position inside/outside asperities
    slip_pos[out_asp_mask] = slip_pos[out_asp_mask]*0.48
    slip_pos[asp_mask] = slip_pos[asp_mask]*0.35
    slip_pos[big_asp_mask] = slip_pos[big_asp_mask]*0.16

    slip_pos = slip_pos/np.max(slip_pos)

    x=lx
    y=lz

    [X,Z] = np.meshgrid(x,y)

    hypo_x = X[np.where(slip_pos==1)[0],np.where(slip_pos==1)[1]]
    hypo_z = Z[np.where(slip_pos==1)[0],np.where(slip_pos==1)[1]]

    return slip_pos, hypo_x, hypo_z
