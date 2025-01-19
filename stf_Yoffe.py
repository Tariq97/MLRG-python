# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
import Yoffe_triangle
from Yoffe_triangle import *


def stf_Yoffe(slip_time,t_onset,Tr_eff,slip_taper,Tacc):


    slip_rate = np.zeros_like(slip_time)

    rupture_onset_time = t_onset

    # Initialize slip_rate to 0 for times before rupture onset
    slip_rate[slip_time < rupture_onset_time] = 0.0
    ik = np.where(slip_time >= rupture_onset_time)[0]
    onslip_time = slip_time[ik] - rupture_onset_time

    # Update slip_rate using the Yoffe_triangle function (to be defined later)
    slip_rate[ik] = Yoffe_triangle(onslip_time, Tr_eff, Tacc) * slip_taper

    slip_rate[slip_rate<0] = 0
    slip_rate = np.nan_to_num(slip_rate)
    slip_rate = np.abs(slip_rate)

    return slip_rate