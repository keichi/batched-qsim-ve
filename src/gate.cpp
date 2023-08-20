#pragma _NEC options "-O4 -finline-functions -report-all"

#include <algorithm>
#include <cmath>

#include "gate.hpp"

void set_zero_state(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                    UINT n)
{
#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << n; i++) {
#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            state_re[sample + i * BATCH_SIZE] = 0;
            state_im[sample + i * BATCH_SIZE] = 0;
            state_re[sample + i * BATCH_SIZE] = 0;
            state_im[sample + i * BATCH_SIZE] = 0;
        }
    }

#pragma omp simd
    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        state_re[sample] = 1;
        state_re[sample] = 1;
    }
}

void apply_single_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                             UINT BATCH_SIZE, UINT n, const double matrix_re[2][2],
                             const double matrix_im[2][2], UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp0_re = state_re[sample + i0 * BATCH_SIZE];
            double tmp0_im = state_im[sample + i0 * BATCH_SIZE];
            double tmp1_re = state_re[sample + i1 * BATCH_SIZE];
            double tmp1_im = state_im[sample + i1 * BATCH_SIZE];

            // clang-format off
            state_re[sample + i0 * BATCH_SIZE] =
                matrix_re[0][0] * tmp0_re - matrix_im[0][0] * tmp0_im +
                matrix_re[0][1] * tmp1_re - matrix_im[0][1] * tmp1_im;
            state_im[sample + i0 * BATCH_SIZE] =
                matrix_re[0][0] * tmp0_im + matrix_im[0][0] * tmp0_re +
                matrix_re[0][1] * tmp1_im + matrix_im[0][1] * tmp1_re;

            state_re[sample + i1 * BATCH_SIZE] =
                matrix_re[1][0] * tmp0_re - matrix_im[1][0] * tmp0_im +
                matrix_re[1][1] * tmp1_re - matrix_im[1][1] * tmp1_im;
            state_im[sample + i1 * BATCH_SIZE] =
                matrix_re[1][0] * tmp0_im + matrix_im[1][0] * tmp0_re +
                matrix_re[1][1] * tmp1_im + matrix_im[1][1] * tmp1_re;
            // clang-format on
        }
    }
}

void apply_single_qubit_gate_soa1(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                                  const std::complex<double> matrix[2][2], UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            std::complex<double> tmp0 = state[sample + i0 * BATCH_SIZE];
            std::complex<double> tmp1 = state[sample + i1 * BATCH_SIZE];

            state[sample + i0 * BATCH_SIZE] = matrix[0][0] * tmp0 + matrix[0][1] * tmp1;
            state[sample + i1 * BATCH_SIZE] = matrix[1][0] * tmp0 + matrix[1][1] * tmp1;
        }
    }
}

#if 0
// Strided implementation
void apply_single_qubit_gate_aos(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                                 const std::complex<double> matrix[2][2], UINT target)
{
    ITYPE mask = 1ULL << target;

#pragma omp parallel for
    for (int sample = 0; sample < BATCH_SIZE; sample++) {

        for (ITYPE i = 0; i < mask; i++) {
#pragma omp simd
            for (ITYPE i0 = i; i0 < 1 << n; i0 += (mask << 1)) {
                ITYPE i1 = i0 + mask;

                std::complex<double> tmp0 = state[i0 + sample * (1 << n)];
                std::complex<double> tmp1 = state[i1 + sample * (1 << n)];

                state[i0 + sample * (1 << n)] = matrix[0][0] * tmp0 + matrix[0][1] * tmp1;
                state[i1 + sample * (1 << n)] = matrix[1][0] * tmp0 + matrix[1][1] * tmp1;
            }
        }
    }
}
#endif
#if 0
// Contiguous implementation
void apply_single_qubit_gate_aos(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                                 const std::complex<double> matrix[2][2], UINT target)
{
    ITYPE mask = 1ULL << target;

#pragma omp parallel for
    for (int sample = 0; sample < BATCH_SIZE; sample++) {

        for (ITYPE i = 0; i < 1ULL << n; i += (mask << 1)) {
#pragma omp simd
            for (ITYPE i0 = i; i0 < i + mask; i0++) {
                ITYPE i1 = i0 + mask;

                std::complex<double> tmp0 = state[i0 + sample * (1 << n)];
                std::complex<double> tmp1 = state[i1 + sample * (1 << n)];

                state[i0 + sample * (1 << n)] = matrix[0][0] * tmp0 + matrix[0][1] * tmp1;
                state[i1 + sample * (1 << n)] = matrix[1][0] * tmp0 + matrix[1][1] * tmp1;
            }
        }
    }
}
#endif
#if 1
// Gather-Scatter implementation
void apply_single_qubit_gate_aos(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                                 const std::complex<double> matrix[2][2], UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (int sample = 0; sample < BATCH_SIZE; sample++) {

#pragma omp simd
        for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
            ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
            ITYPE i1 = i0 | mask;

            std::complex<double> tmp0 = state[i0 + sample * (1 << n)];
            std::complex<double> tmp1 = state[i1 + sample * (1 << n)];

            state[i0 + sample * (1 << n)] = matrix[0][0] * tmp0 + matrix[0][1] * tmp1;
            state[i1 + sample * (1 << n)] = matrix[1][0] * tmp0 + matrix[1][1] * tmp1;
        }
    }
}
#endif

void apply_two_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                          UINT BATCH_SIZE, UINT n, const double matrix_re[4][4],
                          const double matrix_im[4][4], UINT target, UINT control)
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
    for (ITYPE i = 0; i < 1ULL << (n - 2); i++) {
        ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
        ITYPE i01 = i00 | target_mask;
        ITYPE i10 = i00 | control_mask;
        ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp00_re = state_re[sample + i00 * BATCH_SIZE];
            double tmp00_im = state_im[sample + i00 * BATCH_SIZE];
            double tmp01_re = state_re[sample + i01 * BATCH_SIZE];
            double tmp01_im = state_im[sample + i01 * BATCH_SIZE];
            double tmp10_re = state_re[sample + i10 * BATCH_SIZE];
            double tmp10_im = state_im[sample + i10 * BATCH_SIZE];
            double tmp11_re = state_re[sample + i11 * BATCH_SIZE];
            double tmp11_im = state_im[sample + i11 * BATCH_SIZE];

            // clang-format off
            state_re[sample + i00 * BATCH_SIZE] =
                matrix_re[0][0] * tmp00_re - matrix_im[0][0] * tmp00_im +
                matrix_re[0][1] * tmp01_re - matrix_im[0][1] * tmp01_im +
                matrix_re[0][2] * tmp10_re - matrix_im[0][2] * tmp10_im +
                matrix_re[0][3] * tmp11_re - matrix_im[0][3] * tmp11_im;
            state_im[sample + i00 * BATCH_SIZE] =
                matrix_re[0][0] * tmp00_im + matrix_im[0][0] * tmp00_re +
                matrix_re[0][1] * tmp01_im + matrix_im[0][1] * tmp01_re +
                matrix_re[0][2] * tmp10_im + matrix_im[0][2] * tmp10_re +
                matrix_re[0][3] * tmp11_im + matrix_im[0][3] * tmp11_re;

            state_re[sample + i01 * BATCH_SIZE] =
                matrix_re[1][0] * tmp00_re - matrix_im[1][0] * tmp00_im +
                matrix_re[1][1] * tmp01_re - matrix_im[1][1] * tmp01_im +
                matrix_re[1][2] * tmp10_re - matrix_im[1][2] * tmp10_im +
                matrix_re[1][3] * tmp11_re - matrix_im[1][3] * tmp11_im;
            state_im[sample + i01 * BATCH_SIZE] =
                matrix_re[1][0] * tmp00_im + matrix_im[1][0] * tmp00_re +
                matrix_re[1][1] * tmp01_im + matrix_im[1][1] * tmp01_re +
                matrix_re[1][2] * tmp10_im + matrix_im[1][2] * tmp10_re +
                matrix_re[1][3] * tmp11_im + matrix_im[1][3] * tmp11_re;

            state_re[sample + i10 * BATCH_SIZE] =
                matrix_re[2][0] * tmp00_re - matrix_im[2][0] * tmp00_im +
                matrix_re[2][1] * tmp01_re - matrix_im[2][1] * tmp01_im +
                matrix_re[2][2] * tmp10_re - matrix_im[2][2] * tmp10_im +
                matrix_re[2][3] * tmp11_re - matrix_im[2][3] * tmp11_im;
            state_im[sample + i10 * BATCH_SIZE] =
                matrix_re[2][0] * tmp00_im + matrix_im[2][0] * tmp00_re +
                matrix_re[2][1] * tmp01_im + matrix_im[2][1] * tmp01_re +
                matrix_re[2][2] * tmp10_im + matrix_im[2][2] * tmp10_re +
                matrix_re[2][3] * tmp11_im + matrix_im[2][3] * tmp11_re;

            state_re[sample + i11 * BATCH_SIZE] =
                matrix_re[3][0] * tmp00_re - matrix_im[3][0] * tmp00_im +
                matrix_re[3][1] * tmp01_re - matrix_im[3][1] * tmp01_im +
                matrix_re[3][2] * tmp10_re - matrix_im[3][2] * tmp10_im +
                matrix_re[3][3] * tmp11_re - matrix_im[3][3] * tmp11_im;
            state_im[sample + i11 * BATCH_SIZE] =
                matrix_re[3][0] * tmp00_im + matrix_im[3][0] * tmp00_re +
                matrix_re[3][1] * tmp01_im + matrix_im[3][1] * tmp01_re +
                matrix_re[3][2] * tmp10_im + matrix_im[3][2] * tmp10_re +
                matrix_re[3][3] * tmp11_im + matrix_im[3][3] * tmp11_re;
            // clang-format on
        }
    }
}

void apply_x_gate_opt(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                      UINT n, UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp0_re = state_re[sample + i0 * BATCH_SIZE];
            double tmp0_im = state_im[sample + i0 * BATCH_SIZE];
            double tmp1_re = state_re[sample + i1 * BATCH_SIZE];
            double tmp1_im = state_im[sample + i1 * BATCH_SIZE];

            state_re[sample + i0 * BATCH_SIZE] = tmp1_re;
            state_im[sample + i0 * BATCH_SIZE] = tmp1_im;

            state_re[sample + i1 * BATCH_SIZE] = tmp0_re;
            state_im[sample + i1 * BATCH_SIZE] = tmp0_im;
        }
    }
}

void apply_y_gate_opt(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                      UINT n, UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp0_re = state_re[sample + i0 * BATCH_SIZE];
            double tmp0_im = state_im[sample + i0 * BATCH_SIZE];
            double tmp1_re = state_re[sample + i1 * BATCH_SIZE];
            double tmp1_im = state_im[sample + i1 * BATCH_SIZE];

            state_re[sample + i0 * BATCH_SIZE] = tmp1_im;
            state_im[sample + i0 * BATCH_SIZE] = -tmp1_re;

            state_re[sample + i1 * BATCH_SIZE] = tmp0_im;
            state_im[sample + i1 * BATCH_SIZE] = -tmp0_re;
        }
    }
}

void apply_z_gate_opt(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                      UINT n, UINT target)
{
    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp1_re = state_re[sample + i1 * BATCH_SIZE];
            double tmp1_im = state_im[sample + i1 * BATCH_SIZE];

            state_re[sample + i1 * BATCH_SIZE] = -tmp1_re;
            state_im[sample + i1 * BATCH_SIZE] = -tmp1_im;
        }
    }
}

void apply_h_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                  UINT n, UINT target)
{
    static double inv_sqrt2 = 1 / std::sqrt(2);
    static double matrix_re[2][2] = {{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}};
    static double matrix_im[2][2] = {{0, 0}, {0, 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_rx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, double theta, UINT target)
{
    double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
    double matrix_im[2][2] = {{0, -std::sin(theta / 2)}, {-std::sin(theta / 2), 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_rx_gate_soa1(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                        double theta, UINT target)
{
    std::complex<double> matrix[2][2] = {{std::complex<double>(std::cos(theta / 2), 0),
                                          std::complex<double>(0, -std::sin(theta / 2))},
                                         {std::complex<double>(-std::sin(theta / 2), 0),
                                          std::complex<double>(std::cos(theta / 2), 0)}};

    apply_single_qubit_gate_soa1(state, BATCH_SIZE, n, matrix, target);
}

void apply_rx_gate_aos(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT n,
                       double theta, UINT target)
{
    std::complex<double> matrix[2][2] = {{std::complex<double>(std::cos(theta / 2), 0),
                                          std::complex<double>(0, -std::sin(theta / 2))},
                                         {std::complex<double>(-std::sin(theta / 2), 0),
                                          std::complex<double>(std::cos(theta / 2), 0)}};

    apply_single_qubit_gate_aos(state, BATCH_SIZE, n, matrix, target);
}

void apply_sx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target)
{
    static double matrix_re[2][2] = {{0.5, 0.5}, {0.5, 0.5}};
    static double matrix_im[2][2] = {{0.5, -0.5}, {-0.5, 0.5}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_sy_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target)
{
    static double matrix_re[2][2] = {{0.5, -0.5}, {0.5, 0.5}};
    static double matrix_im[2][2] = {{0.5, -0.5}, {0.5, 0.5}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_sw_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target)
{
    static double inv_sqrt2 = 1 / std::sqrt(2);
    static double matrix_re[2][2] = {{inv_sqrt2, -0.5}, {0.5, inv_sqrt2}};
    static double matrix_im[2][2] = {{0, -0.5}, {-0.5, 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_t_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                  UINT n, UINT target)
{
    static double matrix_re[2][2] = {{1, 0}, {0, std::sqrt(2) / 2}};
    static double matrix_im[2][2] = {{0, 0}, {0, std::sqrt(2) / 2}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_cnot_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                     UINT n, UINT target, UINT control)
{
    static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
    static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target, control);
}

void apply_cnot_gate_opt(std::vector<double> &state_re, std::vector<double> &state_im,
                         UINT BATCH_SIZE, UINT n, UINT target, UINT control)
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
    for (ITYPE i = 0; i < 1ULL << (n - 2); i++) {
        ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
        ITYPE i01 = i00 | target_mask;
        ITYPE i10 = i00 | control_mask;
        ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp10_re = state_re[sample + i10 * BATCH_SIZE];
            double tmp10_im = state_im[sample + i10 * BATCH_SIZE];
            double tmp11_re = state_re[sample + i11 * BATCH_SIZE];
            double tmp11_im = state_im[sample + i11 * BATCH_SIZE];

            state_re[sample + i10 * BATCH_SIZE] = tmp11_re;
            state_im[sample + i10 * BATCH_SIZE] = tmp11_im;

            state_re[sample + i11 * BATCH_SIZE] = tmp10_re;
            state_im[sample + i11 * BATCH_SIZE] = tmp10_im;
        }
    }
}

void apply_cz_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target, UINT control)
{
    static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
    static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target, control);
}

void apply_iswaplike_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                          UINT BATCH_SIZE, UINT n, double theta, UINT target, UINT control)
{
    double matrix_re[4][4] = {
        {1, 0, 0, 0}, {0, std::cos(theta), 0, 0}, {0, 0, std::cos(theta), 0}, {0, 0, 0, 1}};
    double matrix_im[4][4] = {
        {0, 0, 0, 0}, {0, 0, -std::sin(theta), 0}, {0, -std::sin(theta), 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target, control);
}

void apply_cz_gate_opt(std::vector<double> &state_re, std::vector<double> &state_im,
                       UINT BATCH_SIZE, UINT n, UINT target, UINT control)
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
    for (ITYPE i = 0; i < 1ULL << (n - 2); i++) {
        ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
        ITYPE i01 = i00 | target_mask;
        ITYPE i10 = i00 | control_mask;
        ITYPE i11 = i00 | control_mask | target_mask;

#pragma omp simd
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            double tmp11_re = state_re[sample + i11 * BATCH_SIZE];
            double tmp11_im = state_im[sample + i11 * BATCH_SIZE];

            state_re[sample + i11 * BATCH_SIZE] = -tmp11_re;
            state_im[sample + i11 * BATCH_SIZE] = -tmp11_im;
        }
    }
}

void apply_depolarizing_gate_1q(std::vector<double> &state_re, std::vector<double> &state_im,
                                UINT BATCH_SIZE, UINT n, UINT target, double prob)
{
    std::vector<double> dice(n);
    std::vector<double> noisy_samples;

    for (int sample = 0; sample < n; sample++) {
        if (dice[sample] < prob) {
            noisy_samples.push_back(sample);
        }
    }

    ITYPE mask = 1ULL << target;
    ITYPE lo_mask = mask - 1;
    ITYPE hi_mask = ~lo_mask;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE i0 = ((i & hi_mask) << 1) | (i & lo_mask);
        ITYPE i1 = i0 | mask;

#pragma omp simd
        for (int j = 0; j < noisy_samples.size(); j++) {
            int sample = noisy_samples[j];

            double tmp0_re = state_re[sample + i0 * BATCH_SIZE];
            double tmp0_im = state_im[sample + i0 * BATCH_SIZE];
            double tmp1_re = state_re[sample + i1 * BATCH_SIZE];
            double tmp1_im = state_im[sample + i1 * BATCH_SIZE];

            if (dice[sample] < prob / 3.0) {
                // Apply X gate
                state_re[sample + i0 * BATCH_SIZE] = tmp1_re;
                state_im[sample + i0 * BATCH_SIZE] = tmp1_im;

                state_re[sample + i1 * BATCH_SIZE] = tmp0_re;
                state_im[sample + i1 * BATCH_SIZE] = tmp0_im;
            } else if (dice[sample] < prob * 2.0 / 3.0) {
                // Apply Y gate
                state_re[sample + i0 * BATCH_SIZE] = tmp1_im;
                state_im[sample + i0 * BATCH_SIZE] = -tmp1_re;

                state_re[sample + i1 * BATCH_SIZE] = tmp0_im;
                state_im[sample + i1 * BATCH_SIZE] = -tmp0_re;
            } else {
                // Apply Z gate
                state_re[sample + i1 * BATCH_SIZE] = -tmp1_re;
                state_im[sample + i1 * BATCH_SIZE] = -tmp1_im;
            }
        }
    }
}
