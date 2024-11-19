import torch

# Environment class for the Frequency Control Problem
class FrequencyControlEnv:
    def __init__(self, H, D, Rg, Tg, delta_t, action_limits, delta_Pe_func, dim_action):
        """
        Initialize the environment.

        Parameters:
        - H: Inertia constants, tensor of shape (dim_action,)
        - D: Damping constants, tensor of shape (dim_action,)
        - Rg: Droop gains, tensor of shape (dim_action,)
        - Tg: Time constants, tensor of shape (dim_action,)
        - delta_t: Time step size
        - action_limits: Tuple (min_action, max_action), each tensor of shape (dim_action,)
        - delta_Pe_func: Function that returns delta_Pe at given time
        - dim_action: Dimension of the action space (number of generators)
        """
        self.H = H
        self.D = D
        self.Rg = Rg
        self.Tg = Tg
        self.delta_t = delta_t
        self.action_limits = action_limits  # Tuple (min_action, max_action)
        self.delta_Pe_func = delta_Pe_func
        self.dim_action = dim_action
        self.state = None  # State vector [delta_f; delta_Pg], shape (2 * dim_action,)
        self.time = 0.0  # Current time

    def reset(self):
        """
        Reset the environment to initial conditions.

        Returns:
        - state: Initial state vector
        """
        # Initialize delta_f and delta_Pg with small random values
        delta_f_init = torch.zeros(self.dim_action, dtype=torch.float32)
        delta_Pg_init = torch.zeros(self.dim_action, dtype=torch.float32)
        self.state = torch.cat([delta_f_init, delta_Pg_init])
        self.time = 0.0
        return self.state.numpy()

    def step(self, action):
        """
        Take a time step in the environment.

        Parameters:
        - action: Control action delta_Pv, tensor of shape (dim_action,)

        Returns:
        - next_state: Next state vector
        - reward: Reward for the current step
        - done: Whether the episode is done
        - info: Additional information (empty dictionary)
        """
        action = torch.tensor(action, dtype=torch.float32)
        # Clip the action to be within the action limits
        min_action, max_action = self.action_limits
        action = torch.max(torch.min(action, max_action), min_action)

        # Extract current state variables
        delta_f_t = self.state[:self.dim_action]
        delta_Pg_t = self.state[self.dim_action:]

        # Get delta_Pe at current time
        delta_Pe_t = self.delta_Pe_func(self.time)

        # Compute f1 and f2 according to the system dynamics
        f1 = - (self.D / (2 * self.H)) * delta_f_t + (delta_Pg_t - delta_Pe_t) / (2 * self.H)
        f2 = (1 / (self.Rg * self.Tg)) * delta_f_t - delta_Pg_t / self.Tg

        # Compute g1 (g2 is zero as per the dynamics)
        g1 = (1 / (2 * self.H)) * action
        g2 = torch.zeros_like(g1)  # g2 is zero

        # Concatenate f1 and f2, g1 and g2
        f = torch.cat([f1, f2])
        g = torch.cat([g1, g2])

        # Update state variables
        self.state = self.state + (f + g) * self.delta_t
        self.time += self.delta_t

        # Compute reward and check constraints
        reward, done = self.compute_reward_and_constraints(delta_f_t, action)

        return self.state.numpy(), reward.item(), done, {}

    def compute_reward_and_constraints(self, delta_f, action):
        """
        Compute the reward and check constraints.

        Parameters:
        - delta_f: Frequency deviations at current time step, tensor of shape (dim_action,)
        - action: Control action taken, tensor of shape (dim_action,)

        Returns:
        - reward: Reward for the current step
        - done: Whether the episode is done (violated constraints)
        """
        # Define the safety constraints as per problem formulation
        delta_f_nadir = torch.min(delta_f)  # Nadir frequency deviation
        delta_f_bound = -0.8  # Frequency bound (Hz)
        delta_f_stable = -0.5  # Stable frequency limit (Hz)

        # Initialize reward and done flag
        reward = 0.0
        done = False

        # Check constraints
        if delta_f_nadir < delta_f_bound:
            # Constraint c1 violated
            reward = -100
            done = True
        elif torch.any(delta_f < delta_f_stable):
            # Constraint c2 violated
            reward = -100
            done = True
        elif torch.any(torch.abs(delta_f) > abs(delta_f_bound)):
            # Constraint c3 violated
            reward = -100
            done = True
        else:
            # Constraints satisfied
            # Compute reward as per Equation (6)
            m1 = 1.0  # Penalty coefficient for action magnitude
            m2 = 1.0  # Reward coefficient for stability
            m3 = 0.1  # Penalty coefficient for time step

            # Penalty for action magnitude
            reward = -m1 * torch.sum(torch.abs(action))

            # Reward for maintaining frequency within bounds
            reward += m2

            # Penalty for time to encourage quick stabilization
            reward -= m3 * self.time

        return reward, done
