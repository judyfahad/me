def DRL_Experince2(s, a, r, s_n):
    """
    Create an experience list for reinforcement learning.

    Parameters:
    s (any): Current state
    a (any): Action taken
    r (any): Reward received
    s_n (any): Next state

    Returns:
    list: A list containing the state, action, reward, and next state
    """
    # Initialize an empty list to store the experience
    e = []
    
    # Append each element to the list
    e.append(s)    # Append the current state
    e.append(a)    # Append the action taken
    e.append(r)    # Append the reward received
    e.append(s_n)  # Append the next state

    return e


