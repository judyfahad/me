import numpy as np
import pandas as pd
import DISCRETE_RND as DISCRETE_RND

def DRL_env23_AoI_LCFS_ext_E_7(s, a, pres, AoI, AoI_mem, Q_E, Q_D, N):
    dbl = len(np.where(s[1:Q_D+1] != -1)[0])
    
    Ar = [0, 1, 2]            # Number of packets that can arrive
    PA = [0.5, 0.1, 0.4]
    TSP = np.array([
        [0.0997, 0.2218, 0.3132],
        [0.0036, 0.0224, 0.0496],
        [0, 0.0003, 0.0014],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    csv_file_path = 'P_EH.csv'  # Update this path if necessary
    P_EH_df = pd.read_csv(csv_file_path, header=None)
    
    P_EH = P_EH_df.values  # Keep the data in its original float format
    np.set_printoptions(precision=20)

    # Debugging: Print the converted data
    #print("Converted to NumPy Array:\n", P_EH)
    
    row_values = P_EH[Q_E, :]

    # Filter out the zero values
    EH = row_values[row_values != 0]
    
    print (EH)

    #EH = P_EH[Q_E, :]
    
    #EH = np.load('P_EH.npy')[Q_E, :]  # Loading precomputed EH data
    
    E = np.arange(0,Q_E + 1)
    #E = np.arange(EH.size)
    
    num_zeros_needed = len(EH) - len(E)

    # Step 3: Append zeros to E
    if num_zeros_needed > 0:
        E = np.append(E, np.zeros(num_zeros_needed))
    print(E)
    
    for i in range(Q_D):
        if AoI_mem[i] != -1:
            AoI_mem[i] += 1
    
    if a == 1:
        r = 0
        p = 0
    else:
        r = (a - 2) % N + 1
        p = (a - 2) // N + 1
    
    if r > 0 and p > 0:
        tsp = TSP[r-1, p-1]
    else:
        tsp = 0
    
    success = DISCRETE_RND.DISCRETE_RND([1, 0], [tsp, 1-tsp], 1)
    
    if success == 1:
        AoI_recent = AoI_mem[0]
        AoI_mem_temp = AoI_mem[r:Q_D]
        AoI_mem = -np.ones(Q_D)
        AoI_mem[:Q_D-r] = AoI_mem_temp
    else:
        AoI_recent = AoI + 1
    
    redundent_index = np.where(AoI_mem == pres - 1)[0]
    drop_r = len(redundent_index)
    AoI_mem[redundent_index] = -1
    
    arr = DISCRETE_RND.DISCRETE_RND(Ar, PA, 1)
    
    if isinstance(arr, np.ndarray):
        arr = int(arr[0])  # Assuming arr is a single-element array; extract the integer
    drop_o = 0
    if arr > 0:
    
        overflow = AoI_mem[Q_D-arr:Q_D]
        drop_o = len(np.where(overflow != -1)[0])
        AoI_mem_temp = AoI_mem[:Q_D-arr]
        AoI_mem[:arr] = 0
        AoI_mem[arr:Q_D] = AoI_mem_temp
    
    r1 = -min(AoI + 1, AoI_recent)
    r2 = -(drop_r + drop_o)
    
    print("Length of E:", len(E))
    print("Length of EH:", len(EH))
    
    e = DISCRETE_RND.DISCRETE_RND(E, EH, 1)
    
    q_e = min(Q_E, s[0] - p + e)
    #r1 = np.array([-r1])
    #r1 = r1.flatten()
    q_e = np.atleast_1d(q_e)
    AoI_mem = np.atleast_1d(AoI_mem)
    r1 = np.atleast_1d(-r1)  # Negate r1 and ensure it's 1D
    s_next = np.concatenate((q_e, AoI_mem, r1))

    # Convert to column vector
    s_next = s_next.reshape(-1, 1)
    
    
    #s_next = np.concatenate(([q_e], AoI_mem, [-r1]))
    
    return s_next, r1, r2, AoI_mem


