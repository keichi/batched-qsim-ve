#pragma once

#include <complex>
#include <random>
#include <vector>

#ifdef __NEC__
#include <asl.h>
#endif

using UINT = unsigned int;
using ITYPE = unsigned long long;

class State
{
protected:
    std::vector<double> state_re_;
    std::vector<double> state_im_;
    UINT batch_size_;
    UINT n_;

#ifdef __NEC__
    asl_random_t rng_;
#endif
    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;

public:
    State(UINT n, UINT batch_size)
        : state_re_((1ULL << n) * batch_size), state_im_((1ULL << n) * batch_size), n_(n),
          batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
#ifdef __NEC__
        if (!asl_library_is_initialized()) {
            asl_library_initialize();
        }

        asl_random_create(&rng_, ASL_RANDOMMETHOD_MT19937);
#endif
    }

    ~State()
    {
#ifdef __NEC__
        asl_random_destroy(rng_);
#endif
    }

    double re(UINT sample, UINT i) { return state_re_[sample + i * batch_size_]; }

    double im(UINT sample, UINT i) { return state_im_[sample + i * batch_size_]; }

    double get_probability(UINT i);

    void set_zero_state();

    void act_single_qubit_gate(double matrix_re[2][2], double matrix_im[2][2], UINT target);

    void act_single_qubit_gate_soa1(std::complex<double> matrix[2][2], UINT target);

    void act_two_qubit_gate(double matrix_re[4][4], double matrix_im[4][4], UINT target,
                            UINT control);

    void act_x_gate_opt(UINT target);

    void act_y_gate_opt(UINT target);

    void act_z_gate_opt(UINT target);

    void act_h_gate(UINT target);

    void act_rx_gate(double theta, UINT target);

    void act_ry_gate(double theta, UINT target);

    void act_rz_gate(double theta, UINT target);

    void act_sx_gate(UINT target);

    void act_sy_gate(UINT target);

    void act_sw_gate(UINT target);

    void act_cnot_gate_opt(UINT target, UINT control);

    void act_t_gate(UINT target);

    void act_cnot_gate(UINT target, UINT control);

    void act_iswaplike_gate(double theta, UINT target, UINT control);

    void act_cz_gate(UINT target, UINT control);

    void act_cz_gate_opt(UINT target, UINT control);

    void act_depolarizing_gate_1q(UINT target, double prob);

    void act_depolarizing_gate_2q(UINT target, UINT control, double prob);
};
