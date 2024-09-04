import numpy as np
import time
from tensorflow.keras import layers, models
import tensorflow as tf
import DRL_env1_AoI_ext as env
import DRL_Agent2 as agent 
import DRL_epsilon_greedy_strategy2 as strategy
import DRL_env23_AoI_LCFS_ext_E_7 as DRL_env23
import DRL_Experince2 as experince
import DRL_Memory2 as mem
import DRL_Memory_Sample2 as memSamp

# %%%%%%%%%%% This is a DQN template %%%%%%%%%%%%
# clc (not needed in Python)
# close all (not needed in Python)
# clear all (not needed in Python, variables are managed differently)

start_time = time.time()

# N=200000;  % Replay memory size
# M=64;    % Minibatch size
# df=0.99; % Discount factor
# epsilon1=1;
# epsilon_end=0.01;
# epsilon_decay=1e-05; % Last 3 parameters are for epsilon greedy strategy
# lr_d=1; % Learning rate for updating weights of policy_net
# lrar_d=1; % Learning rate for updating average reward estimate
# lr=0.001;
# lrar=0.001;
# target_update=4;   # Number of episodes for updating target_net
# episodes=200;  # Number of episodes
# time_steps=1000; # Number of time steps
# soft=0; # Agent dealing with invalid actions (See DRL_Agent)

N = 200000  # Replay memory size
M = 64  # Minibatch size
df = 0.99  # Discount factor
epsilon1 = 1
epsilon_end = 0.01
epsilon_decay = 4.95e-06  # Last 3 parameters are for epsilon greedy strategy
# lr_d = 1  # Learning rate for updating weights of policy_net
# lrar_d = 1  # Learning rate for updating average reward estimate
lr = 0.0005
lrar = 0.0005
target_update = 4  # Number of episodes for updating target_net
episodes = 400  # Number of episodes
time_steps = 1000  # Number of time steps
soft = 0  # Agent dealing with invalid actions (See DRL_Agent)

AOI = np.zeros(15)
PL = np.zeros(15)
AOIa = np.zeros(15)
PLa = np.zeros(15)
TR1 = np.zeros((15, episodes))
TR2 = np.zeros((15, episodes))
TR = np.zeros((15, episodes))
NET = [[None for _ in range(episodes // target_update + 1)] for _ in range(15)]
NET2 = [None for _ in range(15)]
NET3 = [None for _ in range(15)]

K = 3
Rate_l = 5
PRES = 5
# Q_E = 15
Q_D = 16
c1 = 2
c2 = 1
state_dim = Q_D + 2  # Dimension of states
actions_no = Rate_l * K + 1  # Number of actions (overall)

# Defining the neural network layers
model_layers = [
    layers.InputLayer(input_shape=(state_dim,), name="featureinput"),
    layers.Dense(34, activation='relu', name="fc_1"),  # Faris's paper for width eqn
    layers.ReLU(name="relu_2"),
    layers.Dense(34, activation='relu', name="fc_2"),
    layers.ReLU(name="relu_3"),
    layers.Dense(actions_no, name="fc_3")
]
policy_net1 = tf.keras.Sequential(model_layers)
policy_net1.build()

# Using parfor in Python with multiprocessing (simplified example)
for pres in range(1, 9):

    Q_E = 2 * pres - 1

    epsilon = epsilon1

    Total_Reward1 = np.zeros(episodes)  # This is for storing the total reward after each episode (reset)
    Total_Reward2 = np.zeros(episodes)
    Total_Reward = np.zeros(episodes)
    Average_Reward = np.zeros(episodes)  # This is for storing the estimate average reward after each episode (no reset)
    memory = [None] * N  # Reply memory
    minibatch = [None] * M  # Minibatch

    # %%%%%%%%% Deep neural networks initialization %%%%%%%%%
    averageGrad = []  # for training later on  
    averageSqGrad = []  # for training later on 
    executionEnvironment = "auto"  # to run in GPU or CPU

    # See other types of input layer for multi-dimensional input
    policy_net = policy_net1
    target_net = policy_net

    BTN = target_net
    BPN = policy_net
    BTR = -10000000
    BTR1 = -10000000
    BTR2 = -10000000
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    estimate_average_reward = -(c1 * PRES + c2 / PRES)  # Initializing average cost estimate to maximum possible cost or arbitrarily large (= minimum reward?)

    count = 0  # Increases every time step without reset after episodes

    Net = [None for _ in range(episodes // target_update + 1)]
    count_t = 1  # Increases after each target update
    Net[count_t - 1] = target_net

    for i in range(episodes):

        time_count = 0
        episode_total_reward1 = 0  # Initialize episode's reward
        episode_total_reward2 = 0  # Initialize episode's reward
        episode_total_reward = 0

        AoI = 0
        AoI_mem = -np.ones(Q_D)
        s = np.concatenate(([0], AoI_mem, [AoI]))  # Initial state at the beginning of an episode

        start_time = time.time()

        for j in range(time_steps):

            count += 1
            # lr = 1 / (1 + count * lr_d)
            # lrar = 1 / (1 + count * lrar_d)

            s_dlarray = tf.convert_to_tensor(s, dtype=tf.float32)  # converting s to tf.Tensor

            if soft == 1:
                actions = actions_no  # For soft invalid action elimination
            else:
                actions = env.DRL_env1_AoI_ext(s, Rate_l, K, Q_D)  # env1 gives valid actions for state s

            a = agent.DRL_Agent2(s_dlarray, epsilon, policy_net, actions, soft)  # Action selection
            epsilon = strategy.DRL_epsilon_greedy_strategy2(epsilon, epsilon_end, epsilon_decay)  # epsilon update
            s_next, r1, r2, AoI_mem = DRL_env23.DRL_env23_AoI_LCFS_ext_E_7(s, a, PRES, AoI, AoI_mem, Q_E, Q_D, Rate_l)  # get reward and next state corresponding to selecting action a at state s (cost must be -)
            AoI = -r1

            r = c1 * r1 + c2 * r2

            e = experince.DRL_Experince2(s, a, r, s_next)  # experience at a given time step
            memory = mem.DRL_Memory2(e, count, N, memory)  # Pushing e into reply memory
            episode_total_reward1 += r1  # Updating episode's reward
            episode_total_reward2 += r2
            episode_total_reward += r

            if count >= M:
                minibatch = memSamp.DRL_Memory_Sample2(count, N, M, memory)  # Sampling a minibatch
                S = np.array([minibatch[i][0] for i in range(M)])  # All current states from minibatch
                A = np.array([minibatch[i][1] for i in range(M)])
                R = np.array([minibatch[i][2] for i in range(M)])
                S_Next = np.array([minibatch[i][3] for i in range(M)])
                S_dlarray = tf.convert_to_tensor(S, dtype=tf.float32)
                S_Next_dlarray = tf.convert_to_tensor(S_Next, dtype=tf.float32)

                R_current = policy_net(S_dlarray)  # R of current states
                R_next = target_net(S_Next_dlarray)  # R of next states
                R_target = R_current.numpy()  # Convert to NumPy for easier manipulation
                Target = R - estimate_average_reward + np.max(R_next.numpy(), axis=1)  # Predicted R of selected actions
                R_target[np.arange(M), A] = Target  # Final predicted R

                estimate_average_reward += lrar * np.sum(Target - R_current.numpy()[np.arange(M), A])

                # %%%%%%%%%%%%% Training policy_net %%%%%%%%%%%% 
                # If training on a GPU, then convert input to policy_net to a gpuArray.
                # if (executionEnvironment == "auto" and canUseGPU) or executionEnvironment == "gpu":
                #     S_dlarray = tf.convert_to_tensor(S_dlarray, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    grad, loss = modelGradients(policy_net, S_dlarray, R_target)  # computes loss and gradients as defined in modelGradients
                policy_net = adamupdate(policy_net, grad, averageGrad, averageSqGrad, count - M + 1, lr)  # Single SGD

                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            s = s_next

        Total_Reward1[i] = episode_total_reward1
        Total_Reward2[i] = episode_total_reward2
        Total_Reward[i] = episode_total_reward
        Average_Reward[i] = estimate_average_reward

        if episode_total_reward > BTR:
            BTN = target_net
            BPN = policy_net
            BTR = episode_total_reward
            BTR1 = episode_total_reward1
            BTR2 = episode_total_reward2

        if i % target_update == 0:
            target_net = policy_net  # target_net update
            count_t += 1
            Net[count_t - 1] = target_net

    NET[pres - 1] = Net
    NET2[pres - 1] = BTN
    NET3[pres - 1] = BPN
    AOI[pres - 1] = BTR1 / time_steps
    PL[pres - 1] = BTR2 / time_steps

    aoi = Total_Reward1 / time_steps
    pl = Total_Reward2 / time_steps
    AOIa[pres - 1] = np.sum(aoi[episodes - 5:episodes]) / 5
    PLa[pres - 1] = np.sum(pl[episodes - 5:episodes]) / 5

    TR[pres - 1, :] = Total_Reward
    TR1[pres - 1, :] = Total_Reward1
    TR2[pres - 1, :] = Total_Reward2
    with open('output.txt', 'w') as file:
        # Write the processed data to the new file
        file.write('\n\npres')
        file.write(str(pres))
        file.write("\nAOI:")
        file.write(str(AOI))
        file.write("\nAOIa:")
        file.write(str(AOIa))
        file.write("\nPL:")
        file.write(str(PL))
        file.write("\nPLa:")
        file.write(str(PLa))

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
print("***************FINAL**************")
print("AOI:")
print(AOI)
print("AOIa:")
print(AOIa)
print(NET)
print(NET2)
print(NET3)
print("PL:")
print(PL)
print("PLa:")
print(PLa)
print(TR)
print(TR1)
print(TR2)


