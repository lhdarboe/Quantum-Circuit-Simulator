#%%

import numpy as np

#%%

## try out X,Y,Z,H and T gates applied onto |0> |+> |T> states

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j *np.pi / 4)]])

state_0 = np.array([[1], [0]])  # |0>
state_1 = np.array([[0], [1]])  # |1>
state_plus = np.array([[1], [1]]) / np.sqrt(2)  # |+>
state_T = state_0 + np.exp(1j * np.pi / 4) * state_1  # |T>
# %%
