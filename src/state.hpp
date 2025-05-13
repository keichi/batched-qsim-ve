#pragma once

#include <complex>
#include <memory>
#include <vector>

using UINT = unsigned int;
using ITYPE = unsigned long long;

class State
{
public:
    State(UINT n, UINT batch_size);

    ~State();

    std::complex<double> amplitude(UINT sample, UINT i) const;

    double re(UINT sample, UINT i) const;

    double im(UINT sample, UINT i) const;

    double get_probability(UINT i) const;

    UINT dim() const;

    UINT batch_size() const;

    void set_zero_state();

    void act_x_gate(UINT target);

    void act_y_gate(UINT target);

    void act_z_gate(UINT target);

    void act_h_gate(UINT target);

    void act_rx_gate(double theta, UINT target);

    void act_rx_gate(const std::vector<double> &theta, UINT target);

    void act_ry_gate(double theta, UINT target);

    void act_rz_gate(double theta, UINT target);

    void act_sx_gate(UINT target);

    void act_sy_gate(UINT target);

    void act_sw_gate(UINT target);

    void act_cnot_gate(UINT target, UINT control);

    void act_t_gate(UINT target);

    void act_iswaplike_gate(double theta, UINT target, UINT control);

    void act_cx_gate(UINT target, UINT control);

    void act_cz_gate(UINT target, UINT control);

    void act_depolarizing_gate_1q(UINT target, double prob);

    void act_depolarizing_gate_2q(UINT target, UINT control, double prob);

    void synchronize();

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};
