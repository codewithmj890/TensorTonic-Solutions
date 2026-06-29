import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    
    V_new = V.copy()                    # Don't modify the original array

    # Compute TD error
    td_error = r + gamma * V[s_next] - V[s]

    # Update the value of the current state
    V_new[s] = V[s] + alpha * td_error

    return V_new