# Created by Tariq Anwar Aquib,2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np
import models_STF
from importlib import reload

import stf_Yoffe
reload(stf_Yoffe)
from stf_Yoffe import *

import models_STF
reload(models_STF)
from models_STF import *

def Compute_Yoffe_and_ML_STF(slip,vr,X2,Y2,crack_length,t_onset,Tr_eff,Tacc_ratio,t_end,num_pts):
    
    slip_time = np.linspace(0,t_end,num_pts)
    slip_rate = np.zeros(shape=(slip.shape[0],slip.shape[1],501))
    slip_rate_norm = np.zeros_like(slip_rate)

    Tacc = Tr_eff*Tacc_ratio

    for i in range(0,slip.shape[0]):
        for j in range(0,slip.shape[1]):
            slip_rate[i,j,:] = stf_Yoffe(slip_time,t_onset[i,j],Tr_eff[i,j],slip[i,j],Tacc[i,j])
            if slip_rate[i,j,-1] == 0:   # This is done; because Yoffe is not possible if Tr is very large (more than 20); This is mostly at tapered edges where it very small slip
                slip_rate_norm[i,j,:] = slip_rate[i,j,:]/np.max(np.abs((slip_rate[i,j,:]))) # Can divide by NaN when Yoffe not possible (for very small slip values; neglect them)
            else:
                slip_rate_norm[i,j,:] = np.zeros_like(slip_rate[i,j,:])
                Tr_eff[i,j] = 0 # Make it zero here..
                Tacc[i,j] = 0 

    slip_rate_yoffe = slip_rate
    
    print('Yoffe STFs computed')
    # Machine learning inputs 

    stf2 = np.reshape(slip_rate_norm,newshape=[slip.shape[0]*slip.shape[1],slip_rate_norm.shape[-1]],order='F')

    test_meta_all =  np.concatenate([ np.expand_dims(np.ndarray.flatten(vr,order='F'),axis=1), 
                                np.expand_dims(np.ndarray.flatten(slip/np.max(slip),order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(X2,order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(Y2,order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(crack_length,order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(t_onset,order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(Tr_eff,order='F'),axis=1),
                                np.expand_dims(np.ndarray.flatten(slip,order='F'),axis=1)  ], axis=1
                                )


    path_model = '/home/aquibt/Pseudodynamic/STF/ML_models/trained_models/FNO_stf_norm_all_goodfit'
    batch_size = 500

    pred, input = get_predictions_STF(path_model, batch_size, stf2, test_meta_all, 1)

    print('First ML STFs computed')


    ########### If we want to modify where PSVs are greater than 5m/sec use this ##########################
    ############### And the final step; remove all values that are beyond 5 !!!!!

    pred_extended = np.zeros_like(pred)
            
    increments = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

    pred_new = np.copy(pred)
    tr_new = np.copy(test_meta_all[:,6])

    lim = 4.5   # First lim is set at 4.5 and then gradually increased; This is done to avoid patches of 5m/sec
    k1 = 0
    k_indices = np.array([])
    for increment in increments:
        vmax_5 = np.max(pred_new, axis=1)
        k_5 = np.where(vmax_5 > lim)[0]
        
        #print(len(k_5))

        k_indices = np.append(k_indices,k_5)
        

        if len(k_5)>0:
            #### Compute Yoffe STF at only these k_5 points
            test_norm_extended, test_meta_new = compute_STF_new_points(test_meta_all[k_5,:],increment,tacc_ratio=0.1)

            #print(test_norm_extended.shape)
            #print(test_meta_new.shape)

            pred_extended = get_predictions_STF(path_model, batch_size, test_norm_extended, test_meta_new, increment)
            pred_new[k_5, :] = pred_extended[0][:, :]
            tr_new[k_5] = test_meta_all[k_5, 6] * increment
            lim = lim + 0.1
            #print(lim)
            k1 = k1 + 1


    pred = np.nan_to_num(pred,nan=0)
    pred_new = np.nan_to_num(pred_new,nan=0)

    slip_rate_ml0 = np.reshape(pred,newshape=[slip.shape[0],slip.shape[1],slip_rate_norm.shape[-1]],order='F')
    slip_rate_ml1 = np.reshape(pred_new,newshape=[slip.shape[0],slip.shape[1],slip_rate_norm.shape[-1]],order='F')
    # compute Vmax for pred_new
    Vmax_new1 = np.max(pred_new,axis=1)

    Vmax_new1 = np.transpose(np.reshape(Vmax_new1,newshape=(slip.shape[1],slip.shape[0])))
    Tr_new1 = np.transpose(np.reshape(tr_new,newshape=(slip.shape[1],slip.shape[0])))

    # compute Vmax for pred
    Vmax_new0 = np.max(pred,axis=1)

    Vmax_new0 = np.transpose(np.reshape(Vmax_new0,newshape=(slip.shape[1],slip.shape[0])))
    Tr_new0 = np.transpose(np.reshape(test_meta_all[:, 6],newshape=(slip.shape[1],slip.shape[0])))


    #return slip_time, slip_rate_yoffe, slip_rate_ml, Tr_new, Vmax_new, k_indices, pred,  pred_new
    return slip_time, slip_rate_yoffe, slip_rate_ml0, slip_rate_ml1, Tr_new0, Vmax_new0, Tr_new1, Vmax_new1
