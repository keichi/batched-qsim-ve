#pragma once

#include <complex>
#include <vector>

using UINT = unsigned int;
using ITYPE = unsigned long long;

class State
{
protected:
    std::vector<double> state_re_;
    std::vector<double> state_im_;
    UINT batch_size_;
    UINT n_;

public:
    State(UINT n, UINT batch_size)
        : state_re_((1ULL << n) * batch_size), state_im_((1ULL << n) * batch_size), n_(n),
          batch_size_(batch_size)
    {
    }

    double re(UINT sample, UINT i) { return state_re_[sample + i * batch_size_]; }

    double im(UINT sample, UINT i) { return state_im_[sample + i * batch_size_]; }

    void set_zero_state();

    void apply_single_qubit_gate(double matrix_re[2][2], double matrix_im[2][2], UINT target);

    void apply_single_qubit_gate_soa1(std::complex<double> matrix[2][2], UINT target);

    void apply_two_qubit_gate(double matrix_re[4][4], double matrix_im[4][4], UINT target,
                              UINT control);

    void apply_x_gate_opt(UINT target);

    void apply_y_gate_opt(UINT target);

    void apply_z_gate_opt(UINT target);

    void apply_h_gate(UINT target);

    void apply_rx_gate(double theta, UINT target);

    void apply_rx_gate_soa1(double theta, UINT target);

    void apply_rx_gate_aos(double theta, UINT target);

    void apply_sx_gate(UINT target);

    void apply_sy_gate(UINT target);

    void apply_sw_gate(UINT target);

    void apply_cnot_gate_opt(UINT target, UINT control);

    void apply_t_gate(UINT target);

    void apply_cnot_gate(UINT target, UINT control);

    void apply_iswaplike_gate(double theta, UINT target, UINT control);

    void apply_cz_gate(UINT target, UINT control);

    void apply_cz_gate_opt(UINT target, UINT control);

    void apply_depolarizing_gate_1q(UINT target, double prob);
};
