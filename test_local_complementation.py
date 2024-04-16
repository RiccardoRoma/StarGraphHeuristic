from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_state_city, plot_state_hinton

# initial 4-qubit star state with center on center_qubit
num_qubits = 3
center_qubit = 1
circ0 = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    circ0.h(i)

for i in range(num_qubits):
    if i != center_qubit:
        circ0.cz(center_qubit, i)

print("Figure 1: star state circuit")
print(circ0.draw(output="mpl"))

# local complementation on the center
for i in range(num_qubits):
    if i == center_qubit:
        circ0.sx(i)
        #circ0.rx(np.pi/2,i)
    else:
        #circ0.s(i)
        circ0.rz(-np.pi/2, i)
print("Figure 2: star state circuit with local complementation")
print(circ0.draw(output="mpl"))

# reference circuit, which is a fully connected graph state
circ1 = QuantumCircuit(num_qubits)

for i in range(num_qubits):
    circ1.h(i)
for i in range(num_qubits):
    for j in range(i+1, num_qubits):
        circ1.cz(i,j)

print("Figure 3: fully connected graph state circuit")
print(circ1.draw(output="mpl"))

state_simulator = StatevectorSimulator()
job0 = state_simulator.run(circ0)
state0 = Statevector(job0.result().get_statevector(circ0))

job1 = state_simulator.run(circ1)
state1 = Statevector(job1.result().get_statevector(circ1))

overlap = np.abs(state0.inner(state1))
print("overlap: {}".format(overlap))

plt.show()

# look at density matrix if number of qubits is leq to 4
if num_qubits <= 4:
    plot_state_city(state0, title="state from local complementation circuit",color=['midnightblue', 'crimson'], alpha=0.8)
    plot_state_city(state1, title="state from fully connected circuit",color=['midnightblue', 'crimson'], alpha=0.8)

    plot_state_hinton(state0, title="state from local complementation circuit")
    plot_state_hinton(state1, title="state from fully connected circuit")

    plt.show()
else:
    print("number of qubits is to high to plot density matrices!")