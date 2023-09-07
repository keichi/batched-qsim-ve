#pragma _NEC options "-O4 -finline-functions -report-all"
#pragma _NEC options "-fdiag-inline=0 -fdiag-parallel=0 -fdiag-vector=0"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef __NEC__
#include <asl.h>
#endif

#include "state.hpp"

class State::Impl
{
public:
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

    Impl(UINT n, UINT batch_size)
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

    ~Impl()
    {
#ifdef __NEC__
        asl_random_destroy(rng_);
#endif
    }

    double re(UINT sample, UINT i) { return state_re_[sample + i * batch_size_]; }

    double im(UINT sample, UINT i) { return state_im_[sample + i * batch_size_]; }

    double get_probability(UINT i)
    {
        double prob = 0.0;

        for (UINT sample = 0; sample < batch_size_; sample++) {
            double re = state_re_[sample + i * batch_size_];
            double im = state_im_[sample + i * batch_size_];

            prob += re * re + im * im;
        }

        return prob / batch_size_;
    }

    void set_zero_state()
    {
#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << n_; i++) {
#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                state_re_[sample + i * batch_size_] = 0;
                state_im_[sample + i * batch_size_] = 0;
                state_re_[sample + i * batch_size_] = 0;
                state_im_[sample + i * batch_size_] = 0;
            }
        }

#pragma omp simd
        for (int sample = 0; sample < batch_size_; sample++) {
            state_re_[sample] = 1;
            state_re_[sample] = 1;
        }
    }

    void act_single_qubit_gate(double matrix_re[2][2], double matrix_im[2][2], UINT target)
    {
        ITYPE mask = 1ULL << target;
        ITYPE lo_mask = mask - 1;
        ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                // clang-format off
            state_re_[sample + i0 * batch_size_] =
                matrix_re[0][0] * tmp0_re - matrix_im[0][0] * tmp0_im +
                matrix_re[0][1] * tmp1_re - matrix_im[0][1] * tmp1_im;
            state_im_[sample + i0 * batch_size_] =
                matrix_re[0][0] * tmp0_im + matrix_im[0][0] * tmp0_re +
                matrix_re[0][1] * tmp1_im + matrix_im[0][1] * tmp1_re;

            state_re_[sample + i1 * batch_size_] =
                matrix_re[1][0] * tmp0_re - matrix_im[1][0] * tmp0_im +
                matrix_re[1][1] * tmp1_re - matrix_im[1][1] * tmp1_im;
            state_im_[sample + i1 * batch_size_] =
                matrix_re[1][0] * tmp0_im + matrix_im[1][0] * tmp0_re +
                matrix_re[1][1] * tmp1_im + matrix_im[1][1] * tmp1_re;
                // clang-format on
            }
        }
    }

    void act_two_qubit_gate(double matrix_re[4][4], double matrix_im[4][4], UINT target,
                            UINT control)
    {
        ITYPE target_mask = 1ULL << target;
        ITYPE control_mask = 1ULL << control;

        UINT min_qubit_index = std::min(target, control);
        UINT max_qubit_index = std::max(target, control);
        ITYPE min_qubit_mask = 1ULL << min_qubit_index;
        ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
        ITYPE lo_mask = min_qubit_mask - 1;
        ITYPE mid_mask = (max_qubit_mask - 1) ^ lo_mask;
        ITYPE hi_mask = ~(max_qubit_mask - 1);

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 2); i++) {
            ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
            ITYPE i01 = i00 | target_mask;
            ITYPE i10 = i00 | control_mask;
            ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp00_re = state_re_[sample + i00 * batch_size_];
                double tmp00_im = state_im_[sample + i00 * batch_size_];
                double tmp01_re = state_re_[sample + i01 * batch_size_];
                double tmp01_im = state_im_[sample + i01 * batch_size_];
                double tmp10_re = state_re_[sample + i10 * batch_size_];
                double tmp10_im = state_im_[sample + i10 * batch_size_];
                double tmp11_re = state_re_[sample + i11 * batch_size_];
                double tmp11_im = state_im_[sample + i11 * batch_size_];

                // clang-format off
            state_re_[sample + i00 * batch_size_] =
                matrix_re[0][0] * tmp00_re - matrix_im[0][0] * tmp00_im +
                matrix_re[0][1] * tmp01_re - matrix_im[0][1] * tmp01_im +
                matrix_re[0][2] * tmp10_re - matrix_im[0][2] * tmp10_im +
                matrix_re[0][3] * tmp11_re - matrix_im[0][3] * tmp11_im;
            state_im_[sample + i00 * batch_size_] =
                matrix_re[0][0] * tmp00_im + matrix_im[0][0] * tmp00_re +
                matrix_re[0][1] * tmp01_im + matrix_im[0][1] * tmp01_re +
                matrix_re[0][2] * tmp10_im + matrix_im[0][2] * tmp10_re +
                matrix_re[0][3] * tmp11_im + matrix_im[0][3] * tmp11_re;

            state_re_[sample + i01 * batch_size_] =
                matrix_re[1][0] * tmp00_re - matrix_im[1][0] * tmp00_im +
                matrix_re[1][1] * tmp01_re - matrix_im[1][1] * tmp01_im +
                matrix_re[1][2] * tmp10_re - matrix_im[1][2] * tmp10_im +
                matrix_re[1][3] * tmp11_re - matrix_im[1][3] * tmp11_im;
            state_im_[sample + i01 * batch_size_] =
                matrix_re[1][0] * tmp00_im + matrix_im[1][0] * tmp00_re +
                matrix_re[1][1] * tmp01_im + matrix_im[1][1] * tmp01_re +
                matrix_re[1][2] * tmp10_im + matrix_im[1][2] * tmp10_re +
                matrix_re[1][3] * tmp11_im + matrix_im[1][3] * tmp11_re;

            state_re_[sample + i10 * batch_size_] =
                matrix_re[2][0] * tmp00_re - matrix_im[2][0] * tmp00_im +
                matrix_re[2][1] * tmp01_re - matrix_im[2][1] * tmp01_im +
                matrix_re[2][2] * tmp10_re - matrix_im[2][2] * tmp10_im +
                matrix_re[2][3] * tmp11_re - matrix_im[2][3] * tmp11_im;
            state_im_[sample + i10 * batch_size_] =
                matrix_re[2][0] * tmp00_im + matrix_im[2][0] * tmp00_re +
                matrix_re[2][1] * tmp01_im + matrix_im[2][1] * tmp01_re +
                matrix_re[2][2] * tmp10_im + matrix_im[2][2] * tmp10_re +
                matrix_re[2][3] * tmp11_im + matrix_im[2][3] * tmp11_re;

            state_re_[sample + i11 * batch_size_] =
                matrix_re[3][0] * tmp00_re - matrix_im[3][0] * tmp00_im +
                matrix_re[3][1] * tmp01_re - matrix_im[3][1] * tmp01_im +
                matrix_re[3][2] * tmp10_re - matrix_im[3][2] * tmp10_im +
                matrix_re[3][3] * tmp11_re - matrix_im[3][3] * tmp11_im;
            state_im_[sample + i11 * batch_size_] =
                matrix_re[3][0] * tmp00_im + matrix_im[3][0] * tmp00_re +
                matrix_re[3][1] * tmp01_im + matrix_im[3][1] * tmp01_re +
                matrix_re[3][2] * tmp10_im + matrix_im[3][2] * tmp10_re +
                matrix_re[3][3] * tmp11_im + matrix_im[3][3] * tmp11_re;
                // clang-format on
            }
        }
    }

    void act_x_gate_opt(UINT target)
    {
        ITYPE mask = 1ULL << target;
        ITYPE lo_mask = mask - 1;
        ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = tmp1_re;
                state_im_[sample + i0 * batch_size_] = tmp1_im;

                state_re_[sample + i1 * batch_size_] = tmp0_re;
                state_im_[sample + i1 * batch_size_] = tmp0_im;
            }
        }
    }

    void act_y_gate_opt(UINT target)
    {
        ITYPE mask = 1ULL << target;
        ITYPE lo_mask = mask - 1;
        ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = tmp1_im;
                state_im_[sample + i0 * batch_size_] = -tmp1_re;

                state_re_[sample + i1 * batch_size_] = tmp0_im;
                state_im_[sample + i1 * batch_size_] = -tmp0_re;
            }
        }
    }

    void act_z_gate_opt(UINT target)
    {
        ITYPE mask = 1ULL << target;
        ITYPE lo_mask = mask - 1;
        ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i1 * batch_size_] = -tmp1_re;
                state_im_[sample + i1 * batch_size_] = -tmp1_im;
            }
        }
    }

    void act_h_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static double matrix_re[2][2] = {{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}};
        static double matrix_im[2][2] = {{0, 0}, {0, 0}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_rx_gate(double theta, UINT target)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{0, -std::sin(theta / 2)}, {-std::sin(theta / 2), 0}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_ry_gate(double theta, UINT target)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{0, -std::sin(theta / 2)}, {-std::sin(theta / 2), 0}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_rz_gate(double theta, UINT target)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{-std::sin(theta / 2), 0}, {0, std::sin(theta / 2)}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_sx_gate(UINT target)
    {
        static double matrix_re[2][2] = {{0.5, 0.5}, {0.5, 0.5}};
        static double matrix_im[2][2] = {{0.5, -0.5}, {-0.5, 0.5}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_sy_gate(UINT target)
    {
        static double matrix_re[2][2] = {{0.5, -0.5}, {0.5, 0.5}};
        static double matrix_im[2][2] = {{0.5, -0.5}, {0.5, 0.5}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_sw_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static double matrix_re[2][2] = {{inv_sqrt2, -0.5}, {0.5, inv_sqrt2}};
        static double matrix_im[2][2] = {{0, -0.5}, {-0.5, 0}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_t_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static double matrix_re[2][2] = {{1, 0}, {0, inv_sqrt2}};
        static double matrix_im[2][2] = {{0, 0}, {0, inv_sqrt2}};

        act_single_qubit_gate(matrix_re, matrix_im, target);
    }

    void act_cnot_gate(UINT target, UINT control)
    {
        static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(matrix_re, matrix_im, target, control);
    }

    void act_cnot_gate_opt(UINT target, UINT control)
    {
        ITYPE target_mask = 1ULL << target;
        ITYPE control_mask = 1ULL << control;

        UINT min_qubit_index = std::min(target, control);
        UINT max_qubit_index = std::max(target, control);
        ITYPE min_qubit_mask = 1ULL << min_qubit_index;
        ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
        ITYPE lo_mask = min_qubit_mask - 1;
        ITYPE mid_mask = (max_qubit_mask - 1) ^ lo_mask;
        ITYPE hi_mask = ~(max_qubit_mask - 1);

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 2); i++) {
            ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
            ITYPE i01 = i00 | target_mask;
            ITYPE i10 = i00 | control_mask;
            ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp10_re = state_re_[sample + i10 * batch_size_];
                double tmp10_im = state_im_[sample + i10 * batch_size_];
                double tmp11_re = state_re_[sample + i11 * batch_size_];
                double tmp11_im = state_im_[sample + i11 * batch_size_];

                state_re_[sample + i10 * batch_size_] = tmp11_re;
                state_im_[sample + i10 * batch_size_] = tmp11_im;

                state_re_[sample + i11 * batch_size_] = tmp10_re;
                state_im_[sample + i11 * batch_size_] = tmp10_im;
            }
        }
    }

    void act_cz_gate(UINT target, UINT control)
    {
        static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
        static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(matrix_re, matrix_im, target, control);
    }

    void act_iswaplike_gate(double theta, UINT target, UINT control)
    {
        double matrix_re[4][4] = {
            {1, 0, 0, 0}, {0, std::cos(theta), 0, 0}, {0, 0, std::cos(theta), 0}, {0, 0, 0, 1}};
        double matrix_im[4][4] = {
            {0, 0, 0, 0}, {0, 0, -std::sin(theta), 0}, {0, -std::sin(theta), 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(matrix_re, matrix_im, target, control);
    }

    void act_cz_gate_opt(UINT target, UINT control)
    {
        ITYPE target_mask = 1ULL << target;
        ITYPE control_mask = 1ULL << control;

        UINT min_qubit_index = std::min(target, control);
        UINT max_qubit_index = std::max(target, control);
        ITYPE min_qubit_mask = 1ULL << min_qubit_index;
        ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
        ITYPE lo_mask = min_qubit_mask - 1;
        ITYPE mid_mask = (max_qubit_mask - 1) ^ lo_mask;
        ITYPE hi_mask = ~(max_qubit_mask - 1);

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 2); i++) {
            ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
            ITYPE i01 = i00 | target_mask;
            ITYPE i10 = i00 | control_mask;
            ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                double tmp11_re = state_re_[sample + i11 * batch_size_];
                double tmp11_im = state_im_[sample + i11 * batch_size_];

                state_re_[sample + i11 * batch_size_] = -tmp11_re;
                state_im_[sample + i11 * batch_size_] = -tmp11_im;
            }
        }
    }

    void act_depolarizing_gate_1q(UINT target, double prob)
    {
        std::vector<double> dice(batch_size_);
        std::vector<int> x_samples, y_samples, z_samples;

#ifdef __NEC__
        asl_random_generate_d(rng_, batch_size_, dice.data());
#else
        for (int sample = 0; sample < batch_size_; sample++) {
            dice[sample] = dist_(mt_engine_);
        }
#endif

        for (int sample = 0; sample < batch_size_; sample++) {
            if (dice[sample] < prob / 3.0) {
                x_samples.push_back(sample);
            } else if (dice[sample] < prob * 2.0 / 3.0) {
                y_samples.push_back(sample);
            } else if (dice[sample] < prob) {
                z_samples.push_back(sample);
            }
        }

        UINT n_x_samples = x_samples.size();
        UINT n_y_samples = y_samples.size();
        UINT n_z_samples = z_samples.size();

        ITYPE mask = 1ULL << target;
        ITYPE lo_mask = mask - 1;
        ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

#pragma omp simd
            for (int j = 0; j < n_x_samples; j++) {
                int sample = x_samples[j];

                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = tmp1_re;
                state_im_[sample + i0 * batch_size_] = tmp1_im;
                state_re_[sample + i1 * batch_size_] = tmp0_re;
                state_im_[sample + i1 * batch_size_] = tmp0_im;
            }

#pragma omp simd
            for (int j = 0; j < n_y_samples; j++) {
                int sample = y_samples[j];

                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = tmp1_im;
                state_im_[sample + i0 * batch_size_] = -tmp1_re;
                state_re_[sample + i1 * batch_size_] = tmp0_im;
                state_im_[sample + i1 * batch_size_] = -tmp0_re;
            }

#pragma omp simd
            for (int j = 0; j < n_z_samples; j++) {
                int sample = z_samples[j];

                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i1 * batch_size_] = -tmp1_re;
                state_im_[sample + i1 * batch_size_] = -tmp1_im;
            }
        }
    }

    void act_depolarizing_gate_2q(UINT target, UINT control, double prob)
    {
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
    }
};

State::State(UINT n, UINT batch_size) : impl_(std::make_shared<Impl>(n, batch_size)) {}

State::~State() {}

double State::re(UINT sample, UINT i) { return impl_->re(sample, i); }

double State::im(UINT sample, UINT i) { return impl_->im(sample, i); }

double State::get_probability(UINT i) { return impl_->get_probability(i); }

void State::set_zero_state() { return impl_->set_zero_state(); }

void State::act_x_gate(UINT target) { impl_->act_x_gate_opt(target); }

void State::act_y_gate(UINT target) { impl_->act_x_gate_opt(target); }

void State::act_z_gate(UINT target) { impl_->act_z_gate_opt(target); }

void State::act_h_gate(UINT target) { impl_->act_h_gate(target); }

void State::act_rx_gate(double theta, UINT target) { impl_->act_rx_gate(theta, target); }

void State::act_ry_gate(double theta, UINT target) { impl_->act_ry_gate(theta, target); }

void State::act_rz_gate(double theta, UINT target) { impl_->act_rz_gate(theta, target); }

void State::act_sx_gate(UINT target) { impl_->act_sx_gate(target); }

void State::act_sy_gate(UINT target) { impl_->act_sy_gate(target); }

void State::act_sw_gate(UINT target) { impl_->act_sw_gate(target); }

void State::act_t_gate(UINT target) { impl_->act_t_gate(target); }

void State::act_cnot_gate(UINT target, UINT control) { impl_->act_cnot_gate_opt(target, control); }

void State::act_iswaplike_gate(double theta, UINT target, UINT control)
{

    impl_->act_iswaplike_gate(theta, target, control);
}

void State::act_cz_gate(UINT target, UINT control) { impl_->act_cz_gate_opt(target, control); }

void State::act_depolarizing_gate_1q(UINT target, double prob)
{
    impl_->act_depolarizing_gate_1q(target, prob);
}

void State::act_depolarizing_gate_2q(UINT target, UINT control, double prob)
{
    impl_->act_depolarizing_gate_2q(target, control, prob);
}

void State::synchronize() {}
