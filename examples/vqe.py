import numpy as np
import scipy

from veqsim import State, Observable, PauliOperator


def cost(phi):
    obs = Observable()
    obs.add_operator(PauliOperator(-3.8505 / 2, [0, 1], [0, 0]))
    obs.add_operator(PauliOperator(-0.2288 / 2, [1], [1]))
    obs.add_operator(PauliOperator(-1.0466 / 2, [1], [3]))
    obs.add_operator(PauliOperator(-0.2288 / 2, [0], [1]))
    obs.add_operator(PauliOperator(0.2613 / 2, [0, 1], [1, 1]))
    obs.add_operator(PauliOperator(0.2288 / 2, [0, 1], [1, 3]))
    obs.add_operator(PauliOperator(-1.0466 / 2, [0], [3]))
    obs.add_operator(PauliOperator(0.2288 / 2, [0, 1], [3, 1]))
    obs.add_operator(PauliOperator(0.2356 / 2, [0, 1], [3, 3]))

    noise_rate = 0.001

    state = State(2, 10000)
    state.set_zero_state()
    state.act_rx_gate(phi[0], 0)
    state.act_depolarizing_gate_1q(0, noise_rate)
    state.act_rz_gate(phi[1], 0)
    state.act_depolarizing_gate_1q(0, noise_rate)
    state.act_rx_gate(phi[2], 1)
    state.act_depolarizing_gate_1q(1, noise_rate)
    state.act_rz_gate(phi[3], 1)
    state.act_depolarizing_gate_1q(1, noise_rate)
    state.act_cx_gate(0, 1)
    state.act_rz_gate(phi[4], 1)
    state.act_depolarizing_gate_1q(1, noise_rate)
    state.act_rx_gate(phi[5], 1)
    state.act_depolarizing_gate_1q(1, noise_rate)

    energy = np.mean(np.real(state.observe(obs)))

    return energy


def callback(intermediate_result):
    print(intermediate_result.x)
    print(intermediate_result.fun)


def main():
    init = np.random.rand(6)
    res = scipy.optimize.minimize(cost, init, method="Powell", options={"maxiter": 100},
                                  callback=callback)

    print(res)


if __name__ == "__main__":
    main()

