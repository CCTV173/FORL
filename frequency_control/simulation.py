import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import os

from .environment import FrequencyControlEnv
from .policies import linear_droop_control
from .utils import delta_Pe_function, set_seeds

# Initial Seeds
set_seeds()
data_path = os.path.join(os.path.dirname(__file__), 'data', 'IEEE_39bus_Kron.mat')

# data loading
data = loadmat(data_path)

# extracting parameter
H = torch.tensor(np.asarray(data['Kron_39bus']['H'][0, 0]), dtype=torch.float32).flatten()
Damp = torch.tensor(np.asarray(data['Kron_39bus']['D'][0, 0]), dtype=torch.float32).flatten()
dim_action = H.shape[0]
delta_t = 0.01

D = Damp
Rg = torch.ones(dim_action, dtype=torch.float32) * 0.05
Tg = torch.ones(dim_action, dtype=torch.float32) * 0.5

max_action = torch.ones(dim_action, dtype=torch.float32) * 0.2
min_action = -max_action
action_limits = (min_action, max_action)

# initializing the environment
env = FrequencyControlEnv(
    H=H, D=D, Rg=Rg, Tg=Tg, delta_t=delta_t,
    action_limits=action_limits,
    delta_Pe_func=lambda t: delta_Pe_function(t, dim_action),
    dim_action=dim_action)

SimulationLength = 1000
TimeRecord = np.arange(0, SimulationLength + 1) * env.delta_t

Trajectory = []
Actions = []
Rewards = []

#? why here is a thing needed?
state = env.reset()
Trajectory.append(state)

Kp = torch.ones(dim_action, dtype=torch.float32) * 10.0

for step in range(SimulationLength):
    action = linear_droop_control(state, Kp, dim_action, action_limits)
    next_state, reward, done, _ = env.step(action)
    Trajectory.append(next_state)
    Actions.append(action)
    Rewards.append(reward)
    state = next_state
    if done:
        print(f"Simulation terminated at step {step}")
        break

# tensor to np.array
Trajectory = np.array(Trajectory)
Actions = np.array(Actions)
Rewards = np.array(Rewards)

# plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(TimeRecord[:len(Rewards)], Rewards)
plt.xlabel('Time (s)')
plt.ylabel('Reward')
plt.title('Reward vs Time')

plt.subplot(2, 2, 2)
plt.plot(TimeRecord[:len(Actions)], Actions)
plt.xlabel('Time (s)')
plt.ylabel('Action')
plt.title('Control Action vs Time')

plt.subplot(2, 2, 3)
plt.plot(TimeRecord[:len(Trajectory)], Trajectory[:, :dim_action])
plt.xlabel('Time (s)')
plt.ylabel('Frequency Deviation (Hz)')
plt.title('Frequency Deviation vs Time')

plt.subplot(2, 2, 4)
plt.plot(TimeRecord[:len(Trajectory)], Trajectory[:, dim_action:])
plt.xlabel('Time (s)')
plt.ylabel('Generator Power Output Deviation (pu)')
plt.title('Generator Power Output Deviation vs Time')

plt.tight_layout()
plt.show()