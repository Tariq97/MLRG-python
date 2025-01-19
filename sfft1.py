# Created by Tariq Anwar Aquib,2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import scipy.fftpack
import numpy as np

def sfft1(y):

    #dt = t[1] - t[0]                           # dt
    dt = 0.04
    W = 1/dt
    m = (np.shape(y))
  
    if np.remainder(m,2)==1:
        m = m[0] - 1
        y = y[:m]

    f = (W/m)*np.linspace(-m/2,m/2,m)


    # fft
    FT = scipy.fftpack.fft(y)*dt
    FT_shift = scipy.fftpack.fftshift(FT)

    am = np.abs(FT_shift)
    ph = np.angle(FT_shift)

    m1 = int(m/2)
    am = am[m1:]
    f = f[m1:]

    return f, am