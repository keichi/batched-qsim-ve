#pragma _NEC options "-O4 -finline-functions -report-all"

#include <algorithm>
#include <cmath>
// #include <iostream>

#include "gate.hpp"

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

    // std::cout << "control=" << control << " target=" << target << std::endl;

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 2); i++) {
        ITYPE i00 = ((i & hi_mask) << 2) | ((i & mid_mask) << 1) | ((i & lo_mask));
        ITYPE i01 = i00 | target_mask;
        ITYPE i10 = i00 | control_mask;
        ITYPE i11 = i00 | control_mask | target_mask;

        // std::cout << "(i00, i01, i10, i11): " << i00 << ", " << i01 << ", " << i10 << ", " << i11
        //           << std::endl;

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
                matrix_re[0][2] * tmp10_re - matrix_im[0][2] * tmp11_im +
                matrix_re[0][3] * tmp11_re - matrix_im[0][3] * tmp11_im;
            state_im[sample + i00 * BATCH_SIZE] =
                matrix_re[0][0] * tmp00_im + matrix_im[0][0] * tmp00_re +
                matrix_re[0][1] * tmp01_im + matrix_im[0][1] * tmp01_re +
                matrix_re[0][2] * tmp10_im + matrix_im[0][2] * tmp10_re +
                matrix_re[0][3] * tmp11_im + matrix_im[0][3] * tmp11_re;

            state_re[sample + i01 * BATCH_SIZE] =
                matrix_re[1][0] * tmp00_re - matrix_im[1][0] * tmp00_im +
                matrix_re[1][1] * tmp01_re - matrix_im[1][1] * tmp01_im +
                matrix_re[1][2] * tmp10_re - matrix_im[1][2] * tmp11_im +
                matrix_re[1][3] * tmp11_re - matrix_im[1][3] * tmp11_im;
            state_im[sample + i01 * BATCH_SIZE] =
                matrix_re[1][0] * tmp00_im + matrix_im[1][0] * tmp00_re +
                matrix_re[1][1] * tmp01_im + matrix_im[1][1] * tmp01_re +
                matrix_re[1][2] * tmp10_im + matrix_im[1][2] * tmp10_re +
                matrix_re[1][3] * tmp11_im + matrix_im[1][3] * tmp11_re;

            state_re[sample + i10 * BATCH_SIZE] =
                matrix_re[2][0] * tmp00_re - matrix_im[2][0] * tmp00_im +
                matrix_re[2][1] * tmp01_re - matrix_im[2][1] * tmp01_im +
                matrix_re[2][2] * tmp10_re - matrix_im[2][2] * tmp11_im +
                matrix_re[2][3] * tmp11_re - matrix_im[2][3] * tmp11_im;
            state_im[sample + i10 * BATCH_SIZE] =
                matrix_re[2][0] * tmp00_im + matrix_im[2][0] * tmp00_re +
                matrix_re[2][1] * tmp01_im + matrix_im[2][1] * tmp01_re +
                matrix_re[2][2] * tmp10_im + matrix_im[2][2] * tmp10_re +
                matrix_re[2][3] * tmp11_im + matrix_im[2][3] * tmp11_re;

            state_re[sample + i11 * BATCH_SIZE] =
                matrix_re[3][0] * tmp00_re - matrix_im[3][0] * tmp00_im +
                matrix_re[3][1] * tmp01_re - matrix_im[3][1] * tmp01_im +
                matrix_re[3][2] * tmp10_re - matrix_im[3][2] * tmp11_im +
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

void apply_h_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                  UINT n, UINT target)
{
    static double inv_sqrt2 = 1 / std::sqrt(2);
    static double matrix_re[2][2] = {{inv_sqrt2, inv_sqrt2}, {inv_sqrt2, -inv_sqrt2}};
    static double matrix_im[2][2] = {{0, 0}, {0, 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void apply_rx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, double angle, UINT target)
{
    double matrix_re[2][2] = {{std::cos(angle / 2), 0}, {0, std::cos(angle / 2)}};
    double matrix_im[2][2] = {{0, -std::sin(angle / 2)}, {-std::sin(angle / 2), 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
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

void apply_cz_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target, UINT control)
{
    static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
    static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    apply_two_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target, control);
}
