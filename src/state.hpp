#pragma once

#include <complex>
#include <memory>
#include <vector>

#include "observable.hpp"

using UINT = unsigned int;
using ITYPE = unsigned long long;

namespace veqsim
{

class State
{
public:
    State(UINT n, UINT batch_size);

    ~State();

    static void initialize();

    static void finalize();

    std::vector<std::complex<double>> get_vector(UINT sample) const;

    std::complex<double> amplitude(UINT sample, UINT basis) const;

    double re(UINT sample, UINT basis) const;

    double im(UINT sample, UINT basis) const;

    double get_probability(UINT basis) const;

    double get_probability(UINT sample, UINT basis) const;

    UINT dim() const;

    UINT batch_size() const;

    void set_zero_state();

    void act_x_gate(UINT target);

    void act_y_gate(UINT target);

    void act_z_gate(UINT target);

    void act_h_gate(UINT target);

    void act_rx_gate(UINT target, double theta);

    void act_rx_gate(UINT target, const std::vector<double> &theta);

    void act_ry_gate(UINT target, double theta);

    void act_ry_gate(UINT target, const std::vector<double> &theta);

    void act_rz_gate(UINT target, double theta);

    void act_rz_gate(UINT target, const std::vector<double> &theta);

    void act_p_gate(UINT target, double theta);

    void act_p_gate(UINT target, const std::vector<double> &theta);

    void act_sx_gate(UINT target);

    void act_sy_gate(UINT target);

    void act_sw_gate(UINT target);

    void act_cnot_gate(UINT control, UINT target);

    void act_t_gate(UINT target);

    void act_iswaplike_gate(UINT control, UINT target, double theta);

    void act_cx_gate(UINT control, UINT target);

    void act_cz_gate(UINT control, UINT target);

    void act_depolarizing_gate_1q(UINT target, double prob);

    void act_depolarizing_gate_2q(UINT control, UINT target, double prob);

    std::vector<std::complex<double>> observe(const Observable &obs) const;

    void synchronize();

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

void initialize();

void finalize();

}
