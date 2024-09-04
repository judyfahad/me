import random
import numpy as np

def DRL_Memory_Sample2(count, N, M, memory):
    """
    count: current count of experiences
    N: memory size
    M: number of samples to draw
    memory: list of stored experiences
    """
    if count < N:
        memory_array = np.array(memory[:count], dtype=object)
    else:
        memory_array = np.array(memory, dtype=object)
    
    ind = np.random.choice(len(memory_array), M, replace=False)
    minibatch = [memory_array[i] for i in ind]  # Extract the minibatch using the sampled indices
    return minibatch

