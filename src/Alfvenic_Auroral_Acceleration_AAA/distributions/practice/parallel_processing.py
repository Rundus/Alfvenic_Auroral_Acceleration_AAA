import math
import numpy as np
from timebudget import timebudget
import multiprocessing as mp

# --- Create the storage array ---
N = 5 # number of rows
M = 5 # number of columns

mp_arrays = mp.Array('d', N*M)
arr = np.frombuffer(mp_arrays.get_obj())
b = arr.reshape((N,M))

print(b)

# --- define the parallelization function ---

def complex_operation(idx):
    # print("Complex operation. Input index: {:2d}\n".format(input_index))
    iterations_count = round(1e7)
    [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]
    for kdx in range(M):
        b[idx][kdx] = idx/5234234

# # --- Parallel Process ---
processes_count = 32
pool_object = mp.Pool(processes_count)
inputs = range(N)
pool_object.map(complex_operation,inputs)
print(b)



