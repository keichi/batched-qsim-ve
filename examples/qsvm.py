import time
from itertools import combinations

import numpy as np
from veqsim import State


def quantum_kernel(x, y):
    n_repeat = 1
    n_qubit = x.shape[1]
    state = State(n_qubit, x.shape[0])

    state.set_zero_state()

    # ZZ feature map
    for rep in range(n_repeat):
        for i in range(n_qubit):
            state.act_h_gate(i)

        for i, j in combinations(range(n_qubit), 2):
            state.act_p_gate(i, 2 * x[:, i])
            state.act_cnot_gate(i, j)
            state.act_p_gate(j, 2 * (np.pi - x[:, i]) * (np.pi - x[:, j]))
            state.act_cnot_gate(i, j)

    # Conjugate transpose of ZZ feature map
    for rep in range(n_repeat):
        for i, j in combinations(range(n_qubit), 2):
            state.act_cnot_gate(i, j)
            state.act_p_gate(j, -2 * (np.pi - y[:, i]) * (np.pi - y[:, j]))
            state.act_cnot_gate(i, j)
            state.act_p_gate(i, -2 * y[:, i])

        for i in range(n_qubit):
            state.act_h_gate(i)

    # Compute probability of |0..0>
    return state.get_probability_batched(0)


def main():
    num_samples = 100000
    num_features = 10

    x = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples, num_features)

    start = time.perf_counter()

    quantum_kernel(x, y)

    stop = time.perf_counter()

    print(stop - start)


if __name__ == "__main__":
    main()
