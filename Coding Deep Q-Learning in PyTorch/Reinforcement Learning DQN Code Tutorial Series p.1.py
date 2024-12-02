from torch import nn
import torch 
import gym 
from collections import deque
import itertools
import numpy as np
import random

# Discount rate
GAMMA=0.99
# Transitions to sample from replay buffer to train dqn
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
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # seems pytorch always waits a batch dimension for operation?
        q_values = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        # detachs() returns a new tensor that will never require gradient
        # item() converts a one-dimensional tensor in a regular python number
        action = max_q_index.detach().item()
        return action



# Create cartpole env
env = gym.make('CartPole-v1')

# Replay buffer standard Deque
replay_buffer = deque(maxlen=BUFFER_SIZE)
# Reward buffer for each episode
reward_buffer = deque([0, 0], maxlen=100)
# Reward for the single current episode 
episode_reward = 0.0



online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

# Optimizing on the online_net parameters
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs, *_ = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, reward, done, info, *_ = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    
    # save transition tuple in buffer
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs, *_ = env.reset()


# Main training loop
obs, *_ = env.reset()

# EPISODES - Infinity
for step in range(130000):
    # Interpolation: 
    # Starts from 100% to 2% chance to get a random action on "epsilon_decay" steps 
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()
    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
        
    new_obs, reward, done, *_ = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs
    
    episode_reward += reward
    if done:
        obs, *_ = env.reset()
        reward_buffer.append(episode_reward)
        episode_reward = 0.0


    # Start Gradient Step

    # Chooses BATCH_SIZE random samples from replay_buffer
    # and converts it to a numpy matrix
    transitions = np.asarray(random.sample(replay_buffer, BATH_SIZE), dtype=object)
    # Gets every column from matrix and converts it to tensor
    obs_t = torch.as_tensor(np.array(list(transitions[:, 0]), dtype=np.float32))
    actions_t = torch.as_tensor(np.array(list(transitions[:, 1]), dtype = np.int64)).unsqueeze(-1)
    rewards_t = torch.as_tensor(np.array(list(transitions[:, 2]), dtype = np.float32)).unsqueeze(-1)
    dones_t = torch.as_tensor(np.array(list(transitions[:, 3]), dtype = np.float32)).unsqueeze(-1)
    new_obs_t = torch.as_tensor(np.array(list(transitions[:, 4]), dtype = np.float32))

    # Compute targets 
    target_q_values = target_net(new_obs_t)
    # Each new_obs_t has many q_values, so we select the max q_value for each new_obs 
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets  = rewards_t + GAMMA * (1-dones_t) * max_target_q_values
    
    # Compute Loss
    q_values = online_net(obs_t)
    
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    # action_q_values current output, targets is the desired output
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient Descent
    # Pytorch accumulates gradients on every backward call, so its standard to clear previous gradient 
    optimizer.zero_grad()
    # Computes gradients
    loss.backward()
    # Applies gradients in online_net parameters
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        
        if np.mean(reward_buffer) >= 400:
            break

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg reward', np.mean(reward_buffer))
    

print('Model trained!')
env.close()

# Test model and render game!
env = gym.make('CartPole-v1', render_mode="human")
obs, *_ = env.reset()
total_reward = 0

while True:
    action = online_net.act(obs)
    obs, reward, done, *_ = env.step(action)
    total_reward+=reward
    env.render()
    if done:
        print('DONE, Total reward:', total_reward)
        total_reward=0
        obs, *_ = env.reset()

