# Created by Hugo Cruz Jimenez, August 2011, KAUST
# http://docs.python.org/library/random.html
# Modified by Tariq Anwar Aquib, Dec 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import numpy as np
import random
import time


def NormSample(*args):

#function [xi,myseed] = NormSample(mu,sigma,N,myseed)
#%
#%  [xi,NUseed] = NormSample(mu,sigma,N,NUseed)
#%  returns a sample xi randomly drawn from a normal
#%  distribution of random variables X with mean mu and
#%  standard deviation sigma.
#%  N is the length of the distribution to be considered,
#%  NUseed can be a 2D-array for seed-values
#%
#%  INPUT:
#%
#%  mu   - mean of random variable X (default: mu = 0)
#%  sigma - standard deviation of X (default: sigma = 1)
#%  N    - length of distribution (default: N = 100)
#%  NUseed- optional: 2D-array of seeds for random number generator
#%
#%  OUTPUT:
#%
#%  xi   - random sample, chosen from a uniform distribution
#%         of length(N) to select the value of xi among all x
#%  NUseed- returns the 'seeds' i.e. the state of the random
#%         number generators (needed in case one wants to
#%         exactly reproduce the results)
#
#%  Written by Martin Mai (mmai@pangea.Stanford.EDU)
#%  10/22/99
#%  ------------------------------------------------
#

    size = len(args)

    if size == 0:
        mu = 0; sigma = 1; n = 100; myseed=None;
    elif size == 1:
        mu=args[0]; sigma = 1; N = 100; myseed = None;
    elif size == 2:
        mu=args[0]; sigma=args[1]; N = 100; myseed=None;
    elif size == 3:
        mu=args[0]; sigma=args[1]; N= args[2]; myseed=None;
    elif size == 4:
        mu=args[0]; sigma=args[1]; N=args[2]; myseed=args[3];

    if myseed == None:
        myseed = np.zeros(2)
        myseed[0] = int(time.time())
        myseed[1] = myseed[0]*int(time.time())

    random.seed(myseed[0])

    X=[]
    for i in range(0,N):
        X.append(random.normalvariate(mu,sigma))

    xi=random.sample(X,1)[0]
    
    return xi,myseed
