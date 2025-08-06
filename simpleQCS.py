#%%

import numpy as np

#%%

## try out X,Y,Z,H and T gates applied onto |0> |+> |T> states

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j *np.pi / 4)]])
I = np.array([[1, 0], [0, 1]])

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
    solution = gate @ state
    return solution

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

# |psi_out > = U_n ... U_1 U_0 |psi_in>
# need to do kronecker product of all gates

# %%
states = [state_plus, state_0, state_1]
def states_creator(states):

    if len(states) == 2:

        state_out_0 = np.kron(states[0], states[1])
        return state_out_0
    
    if len(states) == 1:

        return states[0]
    
    elif len(states) > 2:

        state_out = np.kron(states[0], states[1])  # |+> |0>

        for i in range(2, len(states)):

            state_out = np.kron(state_out, states[i])

        return state_out

state_out = states_creator(states)
print(state_out)
# %%
#combine the gates together:

gates = [X, H, T]

def gate_combiner(gates):
    if len(gates) == 2:
        gate_out_0 = np.kron(gates[0], gates[1])
        return gate_out_0
    
    if len(gates) == 1:
        return gates[0]
    
    elif len(gates) > 2:
        gate_out = np.kron(gates[0], gates[1])  # X Y

        for i in range(2, len(gates)):
            gate_out = np.kron(gate_out, gates[i])

        return gate_out
    
gate_out = gate_combiner(gates)
print(gate_out)
#%% Measurement 

def measure(state):
    rng = np.random.default_rng()
# %%
solution = gate_out @ state_out

print(solution)
# %% 

#%% 
# %%

bell_state = np.array([[1], [0], [0], [1]]) / np.sqrt(2)  # |00> + |11>

a,b = 1/np.sqrt(2), 1/np.sqrt(2)

psi = a*state_0 + b*state_1

# create the state |psi> |0>

psi_1 = np.kron(psi, bell_state)  
print(psi_1)

# apply CNOT_12 to psi_1
# %%
def probability_measurement(state):
    
    rng = np.random.default_rng()
    probability = rng.choice(state, size = 2)

    return probability

probability_measurement(state_0)
# %%

psi_c, phi_A, phi_B = state_1, state_0, state_1 

flip_gate = np.array([[0, 1], [1, 0]])

def Quantum_Teleportation_Circuit(psi_c, phi_A, phi_B, CNOT, H, I):

    states = [psi_c, phi_A, phi_B]
    psi_in = states_creator(states)

    gates = [H, I, CNOT]

    first_action = gate_combiner([CNOT, I])

    second_action = gate_combiner([H, I, I])
    
    psi_1 = first_action @ psi_in 
    psi_2 = second_action @ psi_1

    probability = probability_measurement(psi_2)
    
    return print('input state: ', psi_in), print('output after first action: ', psi_1), print('output after second action: ', psi_2), print('probability measurement: ', probability)

# %%

psi_in = Quantum_Teleportation_Circuit(psi_c, phi_A, phi_B, CNOT, H, I)
# %%
gate_combiner([CNOT, I])
# %%
