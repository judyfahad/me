import numpy as np

def DRL_env1_AoI_ext(s, N, K, Q_D):
    """
    Determine the valid actions based on the current state.

    Parameters:
    s (list or array): Current state
    N (int): Total number of possible actions related to rate
    K (int): Total number of possible actions related to power
    Q_D (int): Length of a segment of the state

    Returns:
    list: Valid actions
    """
    # Calculate the number of valid states (non-negative values in s[1:Q_D+1])
    dbl = len([x for x in s[1:Q_D+1] if x != -1])

    # If s[0] is 0 or no valid states, the only action is no transmission (1)
    if s[0] == 0 or dbl == 0:
        actions = [1]
    else:
        # Calculate the number of valid actions
        actions_no = min(dbl, N) * min(s[0], K) + 1
        
        # Initialize a list of the appropriate size to hold the actions
        actions_no = int(actions_no)
        actions = [0] * actions_no
        
        # First action is no transmission
        actions[0] = 1
        c = 1
        
        # Generate all other actions
         
        for i in range(1, int(min(s[0], K)) + 1):
            for j in range(1, min(dbl, N) + 1):
                c += 1
                actions[c - 1] = (i - 1) * N + j + 1

    return actions

# Example usage:
# state = [1, 1, -1, 1, -1]  # Example state
# N = 3  # Example total number of actions related to rate
# K = 2  # Example total number of actions related to power
# Q_D = 4  # Length of the state segment
# valid_actions = DRL_env1_AoI_ext(state, N, K, Q_D)
# print(valid_actions)
