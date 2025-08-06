#%%

import numpy as np

from IPython.display import display, Latex
# repr latex
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

    state = state.flatten()
    probabilities = np.abs(state) ** 2

    probabilities = probabilities / np.sum(probabilities)

    rng = np.random.default_rng()
    outcome = rng.choice(len(probabilities), p=probabilities)

    
    rng = np.random.default_rng()
    probability = rng.choice(state, size = 2)

    return probability

probability_measurement(state_0)

# %%

probability_measurement(state_00)

# %%

a = 1/np.sqrt(2)
b = 1/np.sqrt(2)

psi_c = a * state_0 + b * state_1  # |+>

phi_A, phi_B =  state_0, state_1 

states_combined = [psi_c, phi_A, phi_B]
gates_combined = [CNOT, H, I]

flip_gate = np.array([[0, 1], [1, 0]])

def Quantum_Teleportation_Circuit(psi_c, phi_A, phi_B, CNOT, H, I):

    states = [psi_c, phi_A, phi_B]
    psi_in = states_creator(states)

    gates = [H, I, CNOT]

    first_action = gate_combiner([CNOT, I])
    second_action = gate_combiner([H, I, I])
    
    psi_1 = first_action @ psi_in 
    psi_2 = second_action @ psi_1

    #probability = probability_measurement(psi_2)
    
    return print('input state: ', psi_in), print('output after first action: ', psi_1), print('output after second action: ', psi_2)

# %%

Quantum_Teleportation_Circuit(psi_c, phi_A, phi_B, CNOT, H, I)

# %%

class QuantumCircuit:
    def __init__(self, gate, state):
        self.gate = gate
        self.state = state

    def Quantum_Teleportation_Circuit(self):
        gate = self.gate
        state = self.state
        psi_in = states_creator(state)

        first_action = gate_combiner([gate[0], gate[2]])
        second_action = gate_combiner([gate[1], gate[2], gate[2]])
        
        psi_1 = first_action @ psi_in 
        psi_2 = second_action @ psi_1

        return psi_2
    
    def __str__(self):
        return f"Applying gate \n {self.gate} \n to state \n {self.state} \n gives the state \n {self.Quantum_Teleportation_Circuit()}"
# %%

print(QuantumCircuit(gate=[CNOT, H, I], state=[psi_c, phi_A, phi_B]))

# %%
secondaction = gate_combiner([H, I, I])
psi_in = states_creator([psi_c, phi_A, phi_B])
firstaction = gate_combiner([CNOT, I])
psi_1 = firstaction @ psi_in
psi_2 = secondaction @ psi_1

print(psi_2)
# %%

class bit8:
    @staticmethod
    def state_000():
        return np.array([1, 0, 0, 0, 0, 0, 0, 0])  
    @staticmethod
    def state_111():
        return np.array([0, 0, 0, 0, 0, 0, 0, 1])
    @staticmethod
    def state_010():
        return np.array([0, 1, 0, 0, 0, 0, 0, 0])
    @staticmethod
    def state_101():
        return np.array([0, 0, 0, 1, 0, 0, 0, 0])
    @staticmethod
    def state_110():
        return np.array([0, 0, 0, 0, 1, 0, 0, 0])
    @staticmethod
    def state_011():
        return np.array([0, 0, 0, 0, 0, 1, 0, 0])
    @staticmethod
    def state_100():
        return np.array([0, 0, 0, 0, 0, 0, 1, 0])     
    @staticmethod
    def state_001():
        return np.array([0, 0, 1, 0, 0, 0, 0, 0])  

bit8.state_0001()

# %%

final_state = QuantumCircuit(gate=[CNOT, H, I], state=[psi_c, phi_A, phi_B])
final_state2 = final_state.Quantum_Teleportation_Circuit()
flatten_final_state = final_state2.flatten()
probability = np.abs(flatten_final_state) ** 2
print("Final state probabilities:", probability)
print(bit8.state_001())

# calculating |<psi|psi_final>|^2

class probability_calculator:
    def __init__(self, initial_state, final_state):
        self.initial_state = initial_state
        self.final_state = final_state

    def calculate_probability(self):
        inner_product = np.vdot(self.initial_state, self.final_state)
        probability = np.abs(inner_product) ** 2
        return probability
    
    def __str__(self):
        return f"Probability of measuring the final state from the initial state: {self.calculate_probability()}"


prob_010 = probability_calculator(bit8.state_010(),probability)
prob_100 = probability_calculator(bit8.state_100(),probability)
prob_001 = probability_calculator(bit8.state_001(),probability)
prob_111 = probability_calculator(bit8.state_111(),probability)
prob_000 = probability_calculator(bit8.state_000(),probability)
prob_011 = probability_calculator(bit8.state_011(),probability)
prob_101 = probability_calculator(bit8.state_101(),probability)
prob_110 = probability_calculator(bit8.state_110(),probability)
"""
list_of_probabilities = [
    prob_000, prob_001, prob_010, prob_011,
    prob_100, prob_101, prob_110, prob_111
]

sum_probabilities = np.sum(list_of_probabilities)
"""
print(probability_calculator(bit8.state_010(),probability))
# %%
print(list_of_probabilities)

# %%
probability_calculator(bit8.state_010(),probability)
# %%

final_state = QuantumCircuit(gate=[CNOT, H, I], state=[psi_c, phi_A, phi_B])
final_state2 = final_state.Quantum_Teleportation_Circuit()
flatten_final_state = final_state2.flatten()
probability = np.abs(flatten_final_state) #** 2
print("Final state probabilities:", probability)
print(bit8.state_001())

# calculating |<psi|psi_final>|^2

class probability_calculator:
    def __init__(self, initial_state, final_state):
        self.initial_state = initial_state
        self.final_state = final_state

    def calculate_probability(self):
        inner_product = np.vdot(self.initial_state, self.final_state)
        probability = np.abs(inner_product) ** 2
        return probability
    
    #def __str__(self):
    #    return f"Probability of measuring the final state from the initial state: {self.calculate_probability()}"
    
    def __float__(self):
        return self.calculate_probability()

prob_010 = float(probability_calculator(bit8.state_010(),probability))
prob_100 = float(probability_calculator(bit8.state_100(),probability))
prob_001 = float(probability_calculator(bit8.state_001(),probability))
prob_111 = float(probability_calculator(bit8.state_111(),probability))
prob_000 = float(probability_calculator(bit8.state_000(),probability))
prob_011 = float(probability_calculator(bit8.state_011(),probability))
prob_101 = float(probability_calculator(bit8.state_101(),probability))
prob_110 = float(probability_calculator(bit8.state_110(),probability))

list_of_probabilities = [
    prob_000, prob_001, prob_010, prob_011,
    prob_100, prob_101, prob_110, prob_111
]

sum_probabilities = np.sum(list_of_probabilities)

print(probability_calculator(bit8.state_010(),probability))
# %%
probability_calculator(bit8.state_010(),probability)
# %%
print(list_of_probabilities)
np.sum(list_of_probabilities)

# %%

#____________________________________________________________________________________
# CHAT CODE 
#____________________________________________________________________________________

# Displaying quantum states in LaTeX format

from IPython.display import display, Latex

# Suppose you want to show the state |00>
display(Latex(r"$|00\rangle$"))

# Or a more general state with coefficients, e.g.:
alpha = 0.707
beta = 0.707
display(Latex(rf"$\psi = {alpha:.2f}|00\rangle + {beta:.2f}|11\rangle$"))

def display_state(bitstring):
    display(Latex(rf"$|{bitstring}\rangle$"))

# Example usage:

display_state("01")

display(Latex(r"$\frac{1}{\sqrt{2}} |0\rangle + \frac{1}{\sqrt{2}} |1\rangle$"))

from IPython.display import display, Latex

class QuantumStateDisplay:
    def __init__(self, coefficients, basis_states):
        """
        coefficients: list of strings (LaTeX strings or numbers as strings)
        basis_states: list of basis state strings, e.g. ["0", "1"]
        """
        self.coefficients = coefficients
        self.basis_states = basis_states
    
    def __str__(self):
        terms = []
        for coef, state in zip(self.coefficients, self.basis_states):
            terms.append(f"{coef} |{state}\\rangle")
        return " + ".join(terms)
    
    def show(self):
        latex_str = "$" + str(self) + "$"
        display(Latex(latex_str))
qs = QuantumStateDisplay(
    coefficients=[r"\frac{1}{\sqrt{2}}", r"\frac{1}{\sqrt{2}}"],
    basis_states=["0", "1"]
)
qs.show()


# %%
