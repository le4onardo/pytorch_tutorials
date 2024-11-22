from torch import nn
import torch 
import gym 
from collections import deque
import itertools
import numpy as np
import random


# Discount rate
GAMMA=0.99 
# Transitions to sample from replay buffer
BATH_SIZE=32
# Maximun number of transitions to store
BUFFER_SIZE=50000
# Mininum buffer size before computing gradient and training
MIN_REPLAY_SIZE=1000
# Initial randomness
EPSILON_START=1.0
# Final randomness 
EPSILON_END=0.02
# Number of steps to change from epsilon_start to epsilon_end
EPSILON_DECAY=10000
# Number of steps to update target parameters with online parameters
TARGET_UPDATE_FREQ=1000

# Actual Q network
class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Number of input neurons
        in_features =  int(
            # np.prod() - Multiples all values in the observation space's shape. For CartPole, this is one dimensional.
            # env.observation_space - a Box with the min and max state's values. 
            np.prod(env.observation_space.shape)
          )
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n) # the final layer is the number of possible actions to take, in CartPole this is 2. 
        )

    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        pass



# Create cartpole env
env = gym.make('CartPole-v0')

# Replay buffer standard Deque
replay_buffer = deque(maxlen=BUFFER_SIZE)
# Reward buffer for each episode
reward_buffer = deque([0, 0], maxlen=100)
# Reward for the single current episode 
episode_reward = 0.0



online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

# Initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, reward, done, info = env.step(action)
    transition = (obs, action, reward, done, new_obs)

    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()


# Main training loop
obs = env.reset()

# Iterate infinetly
for step in itertools.count():
    # Interpolation: 
    # Starts from 100% to 2% random actions on "epsilon_decay" steps 
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()
    if random_sample <= epsilon:

