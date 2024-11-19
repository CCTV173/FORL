import torch
import numpy as np
import random

# Function to generate delta_Pe(t)

def delta_Pe_function(time, dim_action):
    """
    Function to generate delta_Pe at given time.
    Parameters:
    - time: Current time
    Returns:
    - delta_Pe: Power imbalance at current time, tensor of shape (dim_action,)
    """
    
    delta_Pe = torch.zeros(dim_action, dtype=torch.float32)
    if time >= 0.5 and time < 1.0:
        delta_Pe[0] = 0.1  # Example value, adjust as needed
    return delta_Pe

def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)