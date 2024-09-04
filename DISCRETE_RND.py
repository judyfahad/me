import numpy as np

def DISCRETE_RND(state, prob=None, ndraws=1):
    """
    Generate random numbers from a discrete distribution with specified probabilities.

    Parameters:
    state : array-like
        A vector of discrete states.
    prob : array-like, optional
        A vector specifying the probability on each state. If not provided,
        a uniform distribution is assumed.
    ndraws : int, optional
        Number of draws (default is 1).

    Returns:
    sample : array
        Random numbers sampled from the distribution.

    Notes:
    If the prob vector does not sum up to one, the program will normalize it.
    If the prob vector is missing, a discrete uniform distribution is assumed.
    """
    # If prob is not provided, assume a uniform distribution
    if prob is None:
        prob = np.ones(len(state))
        
    # Handle the case where prob is a scalar and greater than 1
    if np.isscalar(prob) and prob > 1:
        ndraws = prob
        prob = np.ones(len(state))
    else:
        # Ensure state and prob vectors match in length
        if len(state) != len(prob):
            raise ValueError('State and probability vector must match in length.')

    # Normalize the probability vector
    cum_prob = np.cumsum(prob)
    cum_prob = cum_prob / cum_prob[-1]

    # Initialize an array to store the indices of the drawn states
    state_index = np.zeros(ndraws, dtype=int)
    for m in range(ndraws):
        # Find the index where the random number falls within the cumulative probability
        state_index[m] = np.searchsorted(cum_prob, np.random.rand())

    # Convert state_index to integer type explicitly
    state_index = state_index.astype(int)

    # Sample the states based on the drawn indices
    sample = np.array(state)[state_index]
    return sample