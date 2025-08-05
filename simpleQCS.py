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
state_T = ( state_0 + np.exp(1j * np.pi / 4) * state_1  )*1/np.sqrt(2)# |T>

states = [state_0, state_plus, state_1, state_T]

#next exercise

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],    
               [0, 0, 1, 0],
               [0, 0, 0, -1]])

state_00 = np.array([[1], [0], [0], [0]])  # |00>
state_plus_0 = np.array([[1], [0], [1], [0]]) / np.sqrt(2)  # |+0>
state_plus_plus = np.array([[1], [1], [1], [1]]) / 2# |++>

states_2 = [state_00, state_plus_0, state_plus_plus]
# %%
def apply_gate(gate, state):
    return np.dot(gate, state)

solution_X = [apply_gate(X, st) for st in states]
solution_Y = [apply_gate(Y, st) for st in states]
solution_Z = [apply_gate(Z, st) for st in states]
solution_H = [apply_gate(H, st) for st in states]
solution_T = [apply_gate(T, st) for st in states]

solution_CNOT = [apply_gate(CNOT, st) for st in states_2]
solution_CZ = [apply_gate(CZ, st) for st in states_2]
# %%
print('states: |0>, |+>, |1>, |T>')
print('states_2: |00>, |+0>, |++>')
# %%
