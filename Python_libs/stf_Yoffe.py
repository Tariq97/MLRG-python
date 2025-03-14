# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
# import Yoffe_triangle
# from Yoffe_triangle import *

def Yoffe_triangle(time_onslip,rise_time_effective,time_acc):
    Ts = time_acc*(1/1.27)
    Tr = rise_time_effective - 2*Ts

    kappa = 2/(np.pi*Tr*Ts*Ts)

    t = time_onslip
    yoffe = np.zeros_like(t)

    if Tr > 2 * Ts:
            
        # Case 1: t < 0
        yoffe[t < 0] = 0.0
        
        # Case 2: 0 <= t < Ts
        ik = np.where((t >= 0) & (t < Ts))[0]
        C1 = (0.5 * t[ik] + 0.25 * Tr) * np.sqrt(t[ik] * (Tr - t[ik])) + \
            (t[ik] * Tr - Tr**2) * np.arcsin(np.sqrt(t[ik] / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik]) / t[ik]))
        C2 = (3 / 8) * np.pi * Tr**2
        yoffe[ik] = C1 + C2

        del ik, C1
        
        # Case 3: Ts <= t < 2*Ts
        ik = np.where((t >= Ts) & (t < 2 * Ts))[0]
        C1 = (0.5 * t[ik] + 0.25 * Tr) * np.sqrt(t[ik] * (Tr - t[ik])) + \
            (t[ik] * Tr - Tr**2) * np.arcsin(np.sqrt(t[ik] / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik]) / t[ik]))
        C2 = (3 / 8) * np.pi * Tr**2
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        yoffe[ik] = C1 - C2 + C3
        
        del ik, C1, C3
        # Case 4: 2*Ts <= t < Tr
        ik = np.where((t >= 2 * Ts) & (t < Tr))[0]
        C1 = (0.5 * t[ik] + 0.25 * Tr) * np.sqrt(t[ik] * (Tr - t[ik])) + \
            (t[ik] * Tr - Tr**2) * np.arcsin(np.sqrt(t[ik] / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik]) / t[ik]))
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        C4 = (-Ts + 0.5 * t[ik] + 0.25 * Tr) * np.sqrt((t[ik] - 2 * Ts) * (Tr - t[ik] + 2 * Ts)) + \
            Tr * (-Tr + t[ik] - 2 * Ts) * np.arcsin(np.sqrt((t[ik] - 2 * Ts) / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + 2 * Ts) / (t[ik] - 2 * Ts)))
        yoffe[ik] = C1 + C3 + C4
        
        del ik, C3, C4
        # Case 5: Tr <= t < Tr + Ts
        ik = np.where((t >= Tr) & (t < Tr + Ts))[0]
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        C4 = (-Ts + 0.5 * t[ik] + 0.25 * Tr) * np.sqrt((t[ik] - 2 * Ts) * (Tr - t[ik] + 2 * Ts)) + \
            Tr * (-Tr + t[ik] - 2 * Ts) * np.arcsin(np.sqrt((t[ik] - 2 * Ts) / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + 2 * Ts) / (t[ik] - 2 * Ts)))
        C5 = (np.pi / 2) * Tr * (t[ik] - Tr)
        yoffe[ik] = C3 + C4 + C5
        
        del ik, C4
        # Case 6: Tr + Ts <= t < Tr + 2*Ts
        ik = np.where((t >= Tr + Ts) & (t < Tr + 2 * Ts))[0]
        C4 = (-Ts + 0.5 * t[ik] + 0.25 * Tr) * np.sqrt((t[ik] - 2 * Ts) * (Tr - t[ik] + 2 * Ts)) + \
            Tr * (-Tr + t[ik] - 2 * Ts) * np.arcsin(np.sqrt((t[ik] - 2 * Ts) / Tr)) - \
            (3 / 4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + 2 * Ts) / (t[ik] - 2 * Ts)))
        C6 = (np.pi / 2) * Tr * (2 * Ts - t[ik] + Tr)
        yoffe[ik] = C4 + C6
        
        # Case 7: t >= Tr + 2*Ts
        yoffe[t >= (Tr + 2 * Ts)] = 0.0


    if Ts < Tr < 2 * Ts:
        # Case 1: t < Ts
        ik = np.where((t >= 0) & (t < Ts))
        C1 = (0.5 * t[ik] + 0.25 * Tr) * np.sqrt(t[ik] * (Tr - t[ik])) + \
            (t[ik] * Tr - Tr**2) * np.arcsin(np.sqrt(t[ik] / Tr)) - \
            (3/4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik]) / t[ik]))
        C2 = (3/8) * np.pi * Tr**2
        yoffe[ik] = C1 + C2

        del ik, C1, C2

        # Case 2: Ts <= t < Tr
        ik = np.where((t >= Ts) & (t < Tr))
        C1 = (0.5 * t[ik] + 0.25 * Tr) * np.sqrt(t[ik] * (Tr - t[ik])) + \
            (t[ik] * Tr - Tr**2) * np.arcsin(np.sqrt(t[ik] / Tr)) - \
            (3/4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik]) / t[ik]))
        C2 = (3/8) * np.pi * Tr**2
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        yoffe[ik] = C1 - C2 + C3

        del ik, C1, C2, C3

        # Case 3: Tr <= t < 2*Ts
        ik = np.where((t >= Tr) & (t < 2 * Ts))
        C2 = (3/8) * np.pi * Tr**2
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        C5 = (np.pi / 2) * Tr * (t[ik] - Tr)
        yoffe[ik] = C5 + C3 - C2

        del ik, C3, C5

        # Case 4: 2*Ts <= t < Tr + Ts
        ik = np.where((t >= 2 * Ts) & (t < Tr + Ts))
        C3 = (Ts - t[ik] - 0.5 * Tr) * np.sqrt((t[ik] - Ts) * (Tr - t[ik] + Ts)) + \
            Tr * (2 * Tr - 2 * t[ik] + 2 * Ts) * np.arcsin(np.sqrt((t[ik] - Ts) / Tr)) + \
            1.5 * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + Ts) / (t[ik] - Ts)))
        C4 = (-Ts + 0.5 * t[ik] + 0.25 * Tr) * np.sqrt((t[ik] - 2 * Ts) * (Tr - t[ik] + 2 * Ts)) + \
            Tr * (-Tr + t[ik] - 2 * Ts) * np.arcsin(np.sqrt((t[ik] - 2 * Ts) / Tr)) - \
            (3/4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + 2 * Ts) / (t[ik] - 2 * Ts)))
        C5 = (np.pi / 2) * Tr * (t[ik] - Tr)
        yoffe[ik] = C3 + C4 + C5

        del ik, C3, C4, C5

        # Case 5: Tr + Ts <= t < Tr + 2*Ts
        ik = np.where((t >= Tr + Ts) & (t < Tr + 2 * Ts))
        C4 = (-Ts + 0.5 * t[ik] + 0.25 * Tr) * np.sqrt((t[ik] - 2 * Ts) * (Tr - t[ik] + 2 * Ts)) + \
            Tr * (-Tr + t[ik] - 2 * Ts) * np.arcsin(np.sqrt((t[ik] - 2 * Ts) / Tr)) - \
            (3/4) * (Tr**2) * np.arctan(np.sqrt((Tr - t[ik] + 2 * Ts) / (t[ik] - 2 * Ts)))
        C6 = (np.pi / 2) * Tr * (2 * Ts - t[ik] + Tr)
        yoffe[ik] = C4 + C6

        del ik, C4, C6

        # Case 6: t >= Tr + 2*Ts
        yoffe[t >= Tr + 2 * Ts] = 0

    # Compute slip rate
    slip_rate = kappa * yoffe

    # Tolerance handling for near-zero values
    tol = 1e-6
    slip_rate[(slip_rate < 0) & (slip_rate > -tol)] = 0

    return slip_rate
 

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