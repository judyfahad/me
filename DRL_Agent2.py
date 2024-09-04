
import tensorflow as tf
import numpy as np

def DRL_Agent2(s, eps, policy_net, actions, soft):
    """
    DRL_Agent2 function in Python.
    
    Parameters:
    s : current state 
    eps : from strategy
    policy_net : policy network
    actions : either number of actions (soft) or action index/indices (hard)
    soft : soft=1, hard=0
    
    Returns:
    a : selected action
    """

    rate = np.random.rand()
    s_dlarray = tf.convert_to_tensor(s, dtype=tf.float32)  # Convert state to TensorFlow tensor

    # Reshape s_dlarray to ensure it has a batch dimension (1, input_dim)
    s_dlarray = tf.expand_dims(s_dlarray, axis=0)

    b = policy_net(s_dlarray)  # Forward pass through the policy network
    if soft == 1:
        if eps >= rate:
            a = np.random.choice(actions)  # Select random action
        else:
            m = tf.reduce_max(b)  # Find the maximum value
            c = tf.argmax(b)  # Find the index of the max value
            a = c.numpy()  # Convert TensorFlow tensor to a NumPy array
    else:
        if eps >= rate:
            if len(actions) > 1:
                a = np.random.choice(actions)  # Select random action from actions
            else:
                a = actions[0]  # Use the single available action
        else:
            b_actions = tf.gather(b, actions)  # Select actions from b based on indices
            c = tf.argmax(b_actions)  # Find the index of the max value in the selected actions
            a = actions[c.numpy()]  # Map back to original action index

    return int(a)  # Ensure 'a' is an integer
