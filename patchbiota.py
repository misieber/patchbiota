from argparse import ArgumentParser, ArgumentTypeError
import numpy as np
from random import random, sample, randint
from numpy.random import binomial, choice
from numpy import r_, bincount

def non_negative_int(arg):
    nnint = int(arg)
    if nnint < 0:
        raise ArgumentTypeError(arg + ' < 0, must be non-negative')
    return nnint

def non_negative_float(arg):
    f = float(arg)
    if f < 0:
        raise ArgumentTypeError(arg + ' < 0, must be non-negative')
    return f

def float_0_1(arg):
    f = float(arg)
    if f < 0 or f > 1:
        raise ArgumentTypeError(arg + ', must be between 0 and 1')
    return f

# Parse command line arguments
parser = ArgumentParser(description='Simulate host-microbe meta-population model')
parser.add_argument('-s', dest='n_seasons', type=int, default=100, help='Number of dispersal cycles. Default: 100')
parser.add_argument('-t', dest='n_season_length', type=non_negative_int, default=1000, help='Host development time. Default: 1000')
parser.add_argument('-d', dest='f_cost', type=float_0_1, default=0.15, help='Slow type growth disadvantage. Default: 0.15')
parser.add_argument('-e', dest='n_eggs', type=non_negative_int, default=100, help='Number of offspring per host. Default: 100')
parser.add_argument('-c', dest='n_patch_size', type=non_negative_int, default=10000, help='Patch carrying capacity. Default: 10000')
parser.add_argument('-n', dest='n_patches', type=non_negative_int, default=50, help='Number of patches. Default: 10')
parser.add_argument('-m', dest='n_dispersal_pool_size', type=non_negative_int, default=100, help='Dispersal pool size. Default: 100')
parser.add_argument('-b', dest='n_bottleneck', type=non_negative_int, default=5, help='Bottleneck size. Default: 5')
parser.add_argument('-p', dest='n_micro_pool', type=non_negative_int, default=0, help='Fly-independent dispersal. Default: 0')
parser.add_argument('-o', dest='n_dk', type=non_negative_float, default=0.1, help='Maximal patch decay rate. Default: 0.1')
parser.add_argument('-a', dest='n_alpha', type=non_negative_float, default=0.1, help='Slow type contribution to patch decay rate. Default: 0.1')
args = parser.parse_args()

n_patches = args.n_patches
n_patch_size = args.n_patch_size
n_seasons = args.n_seasons
n_season_length = args.n_season_length
n_eggs = args.n_eggs
n_dispersal_pool_size = args.n_dispersal_pool_size
n_bottleneck = args.n_bottleneck
n_micro_pool = args.n_micro_pool
f_cost = args.f_cost
n_dK = args.n_dk
n_alpha = args.n_alpha

def transition(N_f, N_s):
    
    P_f = 0
    P_s = 0
    # Transition probabilities
    if N_f + N_s < n_patch_size:
        fy = 1.0 - f_cost
        P_f = N_f/(N_f + fy*N_s)
        P_s = fy*N_s/(N_f + fy*N_s)

    # Randomly choose a transition
    ra = random()
    if ra < P_f:
        return 0, 1
    elif ra < P_f + P_s:
        return 1, 1
    else:
        return 0, 0

   
# Initialize patches
patches = np.array([[n_bottleneck, 0, n_eggs] for _ in range(n_patches)])
    
# A single mutant in one patch
patches[0][0] = n_bottleneck - 1
patches[0][1] = 1

for season in range(n_seasons):
    # one season of stochastic population dynamics within patches        
    for t in range(n_season_length):
        for patch in patches:
            N_f = patch[0]
            N_s = patch[1]
            H  = patch[2]
            
            # Microbial growth
            if N_f + N_s > 0:
                s, c = transition(N_f, N_s)
                patch[s] += c

            # Host death
            delta = n_dK*(N_f + n_alpha*N_s)/n_patch_size
            patch[2] -= binomial(H, delta)

    # Mean relative abundance of slow type in patches
    mean_f_patches = 0.0
    if patches[:,0:2].sum() > 0:
        mean_f_patches = 1.0*patches[:,1].sum()/patches[:,0:2].sum()

    # Flies leave old patch with sample of local microbes
    flies = []
    for patch in patches[patches[:,2] > 0]:
        if patch[0:2].sum() > 0:
            flies = flies + [r_[fly, n_eggs] for fly in [bincount(choice(2, n_bottleneck, p=patch[0:2]/patch[0:2].sum()), minlength=2) for _ in range(patch[2])]]
        else:
            flies = flies + [[0, 0, n_eggs] for _ in range(patch[2])]        
    flies = np.array(flies)
    
    # Mean relative abundance of slow type in flies
    mean_f_flies = 0.0
    if len(flies) > 0 and flies[:,0:2].sum() > 0:
        mean_f_flies = 1.0*flies[:,1].sum()/flies[:,0:2].sum()

    # Output current mean relative abundances in patches and flies
    print(season, mean_f_patches, mean_f_flies)
    
    # Stop simulation if one of the microbial types goes to fixation
    if mean_f_patches == 0.0 or mean_f_patches == 1.0:
        break

    # Subsample flies to maximal size of dispersal pool
    if len(flies) > n_dispersal_pool_size:
        flies = flies[sample(range(len(flies)), n_dispersal_pool_size)]

    # Form microbial dispersal pool
    micro_pool = np.empty(0)
    if n_micro_pool > 0 and patches[:,0:2].sum() > 0:
        micro_pool = choice(2, n_micro_pool, p=patches[:,0:2].sum(axis=0)/patches[:,0:2].sum())
    
    # Initialize new patches
    patches = np.zeros([n_patches, 3], dtype=int)
        
    # Fly-independent dispersal
    for micro in micro_pool:
        patches[randint(0, n_patches-1)][micro] += 1

    # Fly-mediated dispersal
    for fly in flies:
        patches[randint(0, n_patches-1)] += fly
