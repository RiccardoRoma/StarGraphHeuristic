import numpy as np
import simulation_utils as utils
from qiskit.circuit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2

# setup backend options
backend_opt = {"backend_str": "fake_fractional", 
               "noise_model_id": 0, 
               "fname_noise_model": "", 
               "noise_model_str": "",
               "coupling_map_id": 0,
               "fname_coupling_map": "",
               "coupling_map_str": "",
               "native_basis_gates_str": "",
               "run_locally": True}

# load IBM Quantum credentials
service = utils.load_ibm_credentials(premium_access=False)

# load backend from passmanager calibration
backend = utils.get_backend(service, backend_opt)

# create circuit
qc = QuantumCircuit(3, 1)

qc.h(0)
qc.h(1)
qc.h(2)
qc.cz(0,1)
qc.rz(np.random.rand()*2*np.pi, 0)
qc.h(0)
qc.measure(0,0)
with qc.if_test((qc.cregs[0], 1)):
    qc.x(1)
qc.cz(1,2)

qc.draw(output="mpl")
plt.show()

# create observable
observable = SparsePauliOp.from_list([("IXX", 1.0)])

# create passmanager
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

# transpile circuit
qc_transp = pm.run(qc)

isa_observable = observable.apply_layout(qc_transp.layout)

estimator = EstimatorV2(mode=backend)

job=estimator.run([(qc_transp, isa_observable)])

result = job.result()
