#pragma _NEC options "-O4 -finline-functions -report-all"

#include <cmath>

#include "gate.hpp"

void apply_single_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                             UINT BATCH_SIZE, UINT n, const double matrix_re[2][2],
                             const double matrix_im[2][2], UINT target)
{
#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
        ITYPE mask = 1ULL << target;
        ITYPE mask_lo = mask - 1;
        ITYPE mask_hi = ~mask_lo;

        ITYPE i0 =  ((i & mask_hi) << 1) | (i & mask_lo);
        ITYPE i1 =  i0 | mask;

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
                          const double matrix_im[4][4], UINT target)
{
#pragma omp parallel for
    for (ITYPE i0 = 0; i0 < 1ULL << (n - 1); i0++) {
        ITYPE i00 = i0 | (1ULL << target);
        ITYPE i01 = i0 | (1ULL << target);
        ITYPE i10 = i0 | (1ULL << target);
        ITYPE i11 = i0 | (1ULL << target);

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

void apply_rx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, double angle, UINT target)
{
    double matrix_re[2][2] = {{std::cos(angle / 2), 0}, {0, std::cos(angle / 2)}};
    double matrix_im[2][2] = {{0, -std::sin(angle / 2)}, {-std::sin(angle / 2), 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}
