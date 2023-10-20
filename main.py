import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from tqdm import trange
from model import Circuit
from utils import find_peaks, bin_traj
from cluster_finding import find_cluster
from solver import Solver

import matplotlib
import plt_config
import matplotlib.pyplot as plt
matplotlib.use('Agg') # for running on server without GUI

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)

try:
    os.mkdir('results')
except FileExistsError:
    pass

try:
    os.mkdir('peaks')
except FileExistsError:
    pass

try:
    os.mkdir('avalanches')
except FileExistsError:
    pass

try:
    os.mkdir('graphs')
except FileExistsError:
    pass

N = 1024
d = 2
# Vs = np.linspace(9.0, 16.0, 36)
Vs = [12.0]
R = 12
noise_strength = 0.001
Cth_factors = np.linspace(0.05, 2.0, 40)
couple_factor = 0.02
width_factor = 1.0
T_base = 325
t_max = int(1e6)
dt = 10

min_dist = 101
if min_dist % 2 == 0:
    min_dist += 1
peak_threshold = 1.5

len_x = 1
len_y = 40

Vs, Cth_factors = np.meshgrid(Vs, Cth_factors)
Vs = torch.tensor(Vs.reshape(-1)).reshape(-1, 1)
Cth_factors = torch.tensor(Cth_factors.reshape(-1)).reshape(-1, 1)
batch = len(Vs)
minibatch = int(524288 / N)
n_minibatch = int(np.ceil(batch / minibatch))

for i in trange(n_minibatch):
    Vi = Vs[i * minibatch: (i + 1) * minibatch]
    Cth_factori = Cth_factors[i * minibatch: (i + 1) * minibatch]

    V = Vi[0, 0]
    Cth_factor = Cth_factori[0, 0]
    n_repeat = 100

    solver = Solver(d, len(Vi), N, V, R, noise_strength, Cth_factor, couple_factor, width_factor,
                    T_base, t_max, dt, n_repeat, peak_threshold, min_dist, len_x, len_y)
    solver.model.set_input(Vi, Cth_factori)
    solver.run()

