# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa


# Get first iteration of Vr and Vmax.
import numpy as np
import models
from models import *
import Slip2Stress
from Slip2Stress import *
import compute_CL
from importlib import reload
reload(compute_CL)
from compute_CL import *


def get_Vr_PSV(slip,hypo_x,hypo_z,lx,lz,samp):

    nz,nx = slip.shape

    hyp_ind_x = np.argmin(np.abs(lx-hypo_x))
    hyp_ind_z = np.argmin(np.abs(lz-hypo_z))

    # compute stress drop
    rake = 0
    sigmaS0, sigmaD0 = Slip2Stress(slip,rake,samp)

    sigmaS0 = sigmaS0/np.max(np.abs(sigmaS0))
    sigmaD0 = sigmaD0/np.max(np.abs(sigmaD0))

    # compute X, Y and crack lengths..

    hyp_ind = np.array([hyp_ind_z, hyp_ind_x])  

    X20, Y20, crack_length0 = compute_CL(slip,hyp_ind,samp)
    
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