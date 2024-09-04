import numpy as np


def DRL_Memory2(e, count, N, memory):
    """
    e: current experience to be added to memory
    count: increase every step without reset
    N: memory size
    memory: memory storage before current step; needs to be initialized as
    a list (i.e. memory=[])
    """

    if not isinstance(e, np.ndarray):
        e = np.array(e, dtype=object)

    b = count % N

    if count <= N:
        if len(memory) < N:
            memory.append(e)
        else:
            memory[count - 1] = e
    else:
        if b != 0:
            memory[b - 1] = e
        else:
            memory[N - 1] = e

    return memory