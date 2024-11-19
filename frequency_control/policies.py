import torch

# Linear droop control policy
def linear_droop_control(state, Kp, dim_action, action_limits):
    """
    Linear droop control policy.

    Parameters:
    - state: Current state vector, numpy array of shape (2 * dim_action,)
    - Kp: Proportional gain, tensor of shape (dim_action,)

    Returns:
    - action: Control action, numpy array of shape (dim_action,)
    """

    delta_f = torch.tensor(state[:dim_action], dtype=torch.float32)
    action = -Kp * delta_f

    action = torch.max(torch.min(action, action_limits[1]), action_limits[0])
    return action.numpy()