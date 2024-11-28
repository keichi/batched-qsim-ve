#include <complex>
#include <iostream>

#include "observable.hpp"
#include "state.hpp"

const int BATCH_SIZE = 1000;

int main(int argc, char *argv[])
{
    Observable observable;
    observable.add_operator(PauliOperator(-3.8505 / 2, {0, 1}, {0, 0}));
    observable.add_operator(PauliOperator(-0.2288 / 2, {1}, {1}));
    observable.add_operator(PauliOperator(-1.0466 / 2, {1}, {3}));
    observable.add_operator(PauliOperator(-0.2288 / 2, {0}, {1}));
    observable.add_operator(PauliOperator(0.2613 / 2, {0, 1}, {1, 1}));
    observable.add_operator(PauliOperator(0.2288 / 2, {0, 1}, {1, 3}));
    observable.add_operator(PauliOperator(-1.0466 / 2, {0}, {3}));
    observable.add_operator(PauliOperator(0.2288 / 2, {0, 1}, {3, 1}));
    observable.add_operator(PauliOperator(0.2356 / 2, {0, 1}, {3, 3}));

    std::vector<double> phi(6);
    for (int i = 0; i < 6; i++) {
        phi[i] = std::atof(argv[i]);
    }

    double error_rate = 0.0001;

    State state(2, BATCH_SIZE);
    state.set_zero_state();
    state.act_depolarizing_gate_1q(0, error_rate);
    state.act_rx_gate(phi[0], 0);
    state.act_rz_gate(phi[1], 0);
    state.act_depolarizing_gate_1q(1, error_rate);
    state.act_rx_gate(phi[2], 1);
    state.act_rz_gate(phi[3], 1);
    state.act_cx_gate(0, 1);
    state.act_depolarizing_gate_1q(1, error_rate);
    state.act_rz_gate(phi[4], 1);
    state.act_rx_gate(phi[5], 1);

    double energy = 0.0;
    std::vector<std::complex<double>> vals = observable.get_expectation(state);
    for (int i = 0; i < BATCH_SIZE; i++) {
        energy += vals[i].real();
    }
    energy /= BATCH_SIZE;

    std::cout << energy << std::endl;

    return 0;
}
