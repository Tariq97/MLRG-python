# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa

import numpy as np

def WaterLevel(G, meth='per', wl=0.1, rfil='n', WLseed=None):
    """
    Applies a water-level operation to an array, setting values below a given threshold to zero.
    Optionally replaces zeros with small random numbers or scaled-down original values.

    Parameters:
    G       - 1D or 2D numpy array.
    meth    - 'per' for wl as percentage of max value in G, 'abs' for an absolute value.
    wl      - Water level (default = 0.1 * max(G)).
    rfil    - Option to fill zeros:
              'y' to fill with small random numbers,
              'd' to scale original values,
              'n' to leave as zeros.
    WLseed  - Seed value for the random number generator (optional).

    Returns:
    Y       - Modified array with the same shape as G.
    WLseed  - Updated seed value (None if unused).
    """
    if meth == 'per':
        g = G > wl * np.max(G)
    elif meth == 'abs':
        g = G > wl
    else:
        raise ValueError("Invalid method. Use 'per' or 'abs'.")

    if rfil == 'y':
        Y = G * g
        d = g == 0
        if WLseed is None:
            WLseed = int(sum(100 * np.array([*np.datetime64('now', 'ms').astype(float)])))
        np.random.seed(WLseed)
        r = np.random.randn(*d.shape)
        r = 0.05 * np.max(Y) * (d * r)
        Y += r
    elif rfil == 'd':
        g = g.astype(float)
        g[g == 0] = 0.25
        Y = G * g
        WLseed = None
    elif rfil == 'n':
        Y = G * g
        WLseed = None
    else:
        raise ValueError("Invalid rfil value. Use 'y', 'd', or 'n'.")

    return Y, WLseed
