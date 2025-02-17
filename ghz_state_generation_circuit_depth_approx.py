import numpy as np
import matplotlib.pyplot as plt

def calculate_circuit_depth_merge(substate_size: int, tot_num_qubits: int, c: int = 2) -> int:
    num_substates = np.ceil(tot_num_qubits/substate_size)

    if num_substates > 1:
        # if we have more than one substate
        #depth_substates = calculate_circuit_depth_grow(substate_size)
        depth_substates = calculate_circuit_depth_grow2(substate_size, )
    else:
        # if we have just one substate which contains all qubits
        depth_substates = calculate_circuit_depth_grow(tot_num_qubits)
    # check if number of substates is even or odd
    # if num_substates % 2 == 0:
    #     s = num_substates
    # else:
    #     s = num_substates + 1
    s=num_substates 

    #depth = depth_substates + (np.ceil(np.log2(s+1))-1)*4
    depth = depth_substates + (np.ceil(np.emath.logn(4,s)))*4
    #depth = depth_substates + (np.log2(s+1)-1)*4
    return depth

def calculate_circuit_depth_grow(tot_num_qubits: int) -> int:
    n = (tot_num_qubits-1)/3
    #depth = 3 + (np.log2(n+1)-1)*2
    height = 2*np.log2(n/2+1)-1
    if np.floor(height) % 2 == 0:
        # np.floor(height) is even
        height = round(height + 3 - np.log2(9), 2)
    #depth = 3 + np.ceil(height)*2
    height = np.ceil(height)
    h2 = np.floor((height-1)/2)
    h1 = height - 1 - h2
    depth = 3 + h2 + h1*2 + 1
    return depth

def calculate_circuit_depth_grow2(tot_num_qubits: int, c:int) -> int:
    n = (tot_num_qubits-1)/3
    arg_log = 1-n*(1-c)
    height = np.emath.logn(c, arg_log) - 1
    height = np.ceil(height)
    depth = 3 + height * c
    return depth

N = np.linspace(12, 400, 388, dtype=int)
print(f"test qubit numbers {N}")

# calculate the resulting circuit depth for merging subgraphs of size 3
depth_merge_S3 = [calculate_circuit_depth_merge(3, num-1) for num in N] # num-1 accounts for initial subgraph size 4
depth_merge_S3 = np.asarray(depth_merge_S3) # convert to numpy array

# calculate the resulting circuit depth for merging subgraphs of size 5
depth_merge_S6 = [calculate_circuit_depth_merge(6, num) for num in N]
depth_merge_S6 = np.asarray(depth_merge_S6) # convert to numpy array

# calculate the resulting circuit depth for merging subgraphs of size 5
depth_merge_S12 = [calculate_circuit_depth_merge(12, num) for num in N]
depth_merge_S12 = np.asarray(depth_merge_S12) # convert to numpy array

# calculate the resulting circuit depth for merging subgraphs of size 5
depth_merge_S18 = [calculate_circuit_depth_merge(18, num) for num in N]
depth_merge_S18 = np.asarray(depth_merge_S18) # convert to numpy array

# calculate the resulting circuit depth for merging subgraphs of size 5
depth_merge_S32 = [calculate_circuit_depth_merge(32, num) for num in N]
depth_merge_S32 = np.asarray(depth_merge_S32) # convert to numpy array

# calculate the resulting circuit depth for growing the GHZ state based on layout
depth_grow = [calculate_circuit_depth_grow(num) for num in N] 
depth_grow = np.asarray(depth_grow) # convert to numpy array

# plot
plt.figure(figsize=(10,6))
plt.plot(N, depth_merge_S3, "x--", label="substate size 3")
#plt.plot(N, depth_merge_S5, "x--", label="substate size 5")
plt.plot(N, depth_merge_S6, "x--", label="substate size 6")
plt.plot(N, depth_merge_S12, "x--", label="substate size 12")
plt.plot(N, depth_merge_S18, "x--", label="substate size 18")
plt.plot(N, depth_merge_S32, "x--", label="substate size 32")
plt.plot(N, depth_grow, "x--", label="no merging")
plt.xlabel("number of qubits")
plt.ylabel("circuit depth")
plt.legend()
#plt.xscale("log")
plt.savefig("ghz_gen_depth_eval_new.pdf", bbox_inches="tight")
#plt.show()

# plot
plt.figure(figsize=(10,6))
plt.plot(N, depth_grow-depth_merge_S3, "x--", label="substate size 3")
#plt.plot(N, depth_merge_S5, "x--", label="substate size 5")
plt.plot(N, depth_grow-depth_merge_S6, "x--", label="substate size 6")
plt.plot(N, depth_grow-depth_merge_S12, "x--", label="substate size 12")
plt.plot(N, depth_grow-depth_merge_S18, "x--", label="substate size 18")
plt.plot(N, depth_grow-depth_merge_S32, "x--", label="substate size 32")
plt.xlabel("number of qubits")
plt.ylabel(r"$\mathrm{depth}_{\mathrm{grow}}-\mathrm{depth}_{\mathrm{merge}}$")
plt.legend()
plt.savefig("ghz_gen_depth_diff_eval_new.pdf", bbox_inches="tight")
#plt.show()

# up to 20 qubits
plt.figure(figsize=(10,6))
plt.plot(N[1:17], depth_merge_S3[1:17], "x--", label="substate size 3")
#plt.plot(N[1:17], depth_merge_S5[1:17], "x--", label="substate size 5")
plt.plot(N[1:17], depth_merge_S6[1:17], "x--", label="substate size 6")
plt.plot(N[1:17], depth_merge_S12[1:17], "x--", label="substate size 12")
plt.plot(N[1:17], depth_grow[1:17], "x--", label="no merging")
plt.xlabel("number of qubits")
plt.ylabel("circuit depth")
plt.legend()
#plt.xscale("log")
plt.savefig("ghz_gen_depth_eval_small_new.pdf", bbox_inches="tight")
plt.show()



