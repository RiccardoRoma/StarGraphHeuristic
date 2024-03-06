from qiskit_ibm_runtime import QiskitRuntimeService

# This is a small code for getting number of qubits and couplinng map from a certain IBM device.
 
service = QiskitRuntimeService(channel="ibm_quantum")

backend= service.backend("ibm_brisbane")
# backend = service.backend("ibmq_qasm_simulator")
num_qubits= backend.num_qubits
coupling_map= backend.coupling_map

print('number of qubits', num_qubits)
print('Coupling map', coupling_map)

# This givees the dynamic properties of every qubit such as readout error
for i in range(num_qubits):
    mp=backend.target["measure"][(i,)]
    print(mp)

