def DRL_epsilon_greedy_strategy2(eps, eps_min, eps_decay):
    """
    Update epsilon for an epsilon-greedy strategy.

    This function should be used to update epsilon at the end of each time step.

    Parameters:
    eps (float): Current epsilon value
    eps_min (float): Minimum epsilon value
    eps_decay (float): Decay factor for epsilon

    Returns:
    float: Updated epsilon value
    """
    # Update epsilon by subtracting the decay factor, but ensure it doesn't fall below the minimum value
    eps = max(eps - eps_decay, eps_min)
    return eps

# Example usage:
# eps = 1.0
# eps_min = 0.1
# eps_decay = 0.01
# updated_eps = DRL_epsilon_greedy_strategy2(eps, eps_min, eps_decay)
