#pragma _NEC options "-O4 -finline-functions -report-all"

#include <algorithm>
#include <cmath>

#include "gate.hpp"

void State::set_zero_state()
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

void State::apply_single_qubit_gate(double matrix_re[2][2], double matrix_im[2][2], UINT target)
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

void State::apply_two_qubit_gate(double matrix_re[4][4], double matrix_im[4][4], UINT target,
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

void State::apply_x_gate_opt(UINT target)
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

void State::apply_y_gate_opt(UINT target)
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

void State::apply_z_gate_opt(UINT target)
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

void State::apply_h_gate(UINT target)
{
    static double inv_sqrt2 = 1 / std::sqrt(2);
    static double matrix_re[2][2] = {{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}};
    static double matrix_im[2][2] = {{0, 0}, {0, 0}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_rx_gate(double theta, UINT target)
{
    double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
    double matrix_im[2][2] = {{0, -std::sin(theta / 2)}, {-std::sin(theta / 2), 0}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_sx_gate(UINT target)
{
    static double matrix_re[2][2] = {{0.5, 0.5}, {0.5, 0.5}};
    static double matrix_im[2][2] = {{0.5, -0.5}, {-0.5, 0.5}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_sy_gate(UINT target)
{
    static double matrix_re[2][2] = {{0.5, -0.5}, {0.5, 0.5}};
    static double matrix_im[2][2] = {{0.5, -0.5}, {0.5, 0.5}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_sw_gate(UINT target)
{
    static double inv_sqrt2 = 1 / std::sqrt(2);
    static double matrix_re[2][2] = {{inv_sqrt2, -0.5}, {0.5, inv_sqrt2}};
    static double matrix_im[2][2] = {{0, -0.5}, {-0.5, 0}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_t_gate(UINT target)
{
    static double matrix_re[2][2] = {{1, 0}, {0, std::sqrt(2) / 2}};
    static double matrix_im[2][2] = {{0, 0}, {0, std::sqrt(2) / 2}};

    apply_single_qubit_gate(matrix_re, matrix_im, target);
}

void State::apply_cnot_gate(UINT target, UINT control)
{
    static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
    static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(matrix_re, matrix_im, target, control);
}

void State::apply_cnot_gate_opt(UINT target, UINT control)
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

void State::apply_cz_gate(UINT target, UINT control)
{
    static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
    static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(matrix_re, matrix_im, target, control);
}

void State::apply_iswaplike_gate(double theta, UINT target, UINT control)
{
    double matrix_re[4][4] = {
        {1, 0, 0, 0}, {0, std::cos(theta), 0, 0}, {0, 0, std::cos(theta), 0}, {0, 0, 0, 1}};
    double matrix_im[4][4] = {
        {0, 0, 0, 0}, {0, 0, -std::sin(theta), 0}, {0, -std::sin(theta), 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(matrix_re, matrix_im, target, control);
}

void State::apply_cz_gate_opt(UINT target, UINT control)
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

void State::apply_depolarizing_gate_1q(UINT target, double prob)
{
    std::vector<double> dice(n_);
    std::vector<double> noisy_samples;

    for (int sample = 0; sample < n_; sample++) {
        if (dice[sample] < prob) {
            noisy_samples.push_back(sample);
        }
    }

    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n_ - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int j = 0; j < noisy_samples.size(); j++) {
            int sample = noisy_samples[j];

            double tmp0_re = state_re_[sample + i0 * batch_size_];
            double tmp0_im = state_im_[sample + i0 * batch_size_];
            double tmp1_re = state_re_[sample + i1 * batch_size_];
            double tmp1_im = state_im_[sample + i1 * batch_size_];

            if (dice[sample] < prob / 3.0) {
                // Apply X gate
                state_re_[sample + i0 * batch_size_] = tmp1_re;
                state_im_[sample + i0 * batch_size_] = tmp1_im;

                state_re_[sample + i1 * batch_size_] = tmp0_re;
                state_im_[sample + i1 * batch_size_] = tmp0_im;
            } else if (dice[sample] < prob * 2.0 / 3.0) {
                // Apply Y gate
                state_re_[sample + i0 * batch_size_] = tmp1_im;
                state_im_[sample + i0 * batch_size_] = -tmp1_re;

                state_re_[sample + i1 * batch_size_] = tmp0_im;
                state_im_[sample + i1 * batch_size_] = -tmp0_re;
            } else {
                // Apply Z gate
                state_re_[sample + i1 * batch_size_] = -tmp1_re;
                state_im_[sample + i1 * batch_size_] = -tmp1_im;
            }
        }
    }
}
