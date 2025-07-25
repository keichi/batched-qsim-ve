#pragma _NEC options "-O4 -finline-functions -report-all"
#pragma _NEC options "-fdiag-inline=0 -fdiag-parallel=0 -fdiag-vector=0"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#ifdef __NEC__
#include <asl.h>
#endif

#include "observable.hpp"
#include "state.hpp"

namespace veqsim
{

std::uint32_t insert_zero_to_basis_index(std::uint32_t basis_index, std::uint32_t insert_index)
{
    std::uint32_t mask = (1ULL << insert_index) - 1;
    std::uint32_t temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

std::array<std::complex<double>, 4> PHASE_90ROT()
{
    return {std::complex<double>(1, 0), std::complex<double>(0, 1), std::complex<double>(-1, 0),
            std::complex<double>(0, -1)};
}

class State::Impl
{
public:
    Impl(UINT n, UINT batch_size)
        : state_re_((1ULL << n) * batch_size), state_im_((1ULL << n) * batch_size), n_(n),
          batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
    }

    ~Impl() {}

    static void initialize()
    {
#ifdef __NEC__
        if (!asl_library_is_initialized()) {
            asl_library_initialize();
        }

        asl_random_create(&rng_, ASL_RANDOMMETHOD_MT19937);
#endif
    }

    static void finalize()
    {
#ifdef __NEC__
        asl_random_destroy(rng_);
#endif
    }

    std::vector<std::complex<double>> get_vector(UINT sample) const
    {
        std::vector<std::complex<double>> sv;

        for (ITYPE i = 0; i < 1ULL << n_; i++) {
            sv.emplace_back(state_re_[sample + i * batch_size_],
                            state_im_[sample + i * batch_size_]);
        }

        return sv;
    }

    std::complex<double> amplitude(UINT sample, UINT i) const
    {
        return std::complex(state_re_[sample + i * batch_size_],
                            state_im_[sample + i * batch_size_]);
    }

    double re(UINT sample, UINT i) const { return state_re_[sample + i * batch_size_]; }

    double im(UINT sample, UINT i) const { return state_im_[sample + i * batch_size_]; }

    double get_probability(UINT i) const
    {
        double prob = 0.0;

        for (UINT sample = 0; sample < batch_size_; sample++) {
            double re = state_re_[sample + i * batch_size_];
            double im = state_im_[sample + i * batch_size_];

            prob += re * re + im * im;
        }

        return prob / batch_size_;
    }

    double get_probability(UINT sample, UINT i) const
    {
        double re = state_re_[sample + i * batch_size_];
        double im = state_im_[sample + i * batch_size_];

        return re * re + im * im;
    }

    std::vector<double> get_probability_batched(UINT i) const
    {
        std::vector<double> probs(batch_size_);

        for (UINT sample = 0; sample < batch_size_; sample++) {
            double re = state_re_[sample + i * batch_size_];
            double im = state_im_[sample + i * batch_size_];

            probs[sample] = re * re + im * im;
        }

        return probs;
    }

    UINT dim() const { return 1ULL << n_; }

    UINT batch_size() const { return batch_size_; }

    void set_zero_state()
    {
#pragma omp parallel for
        for (ITYPE i = 0; i < 1ULL << n_; i++) {
#pragma omp simd
            for (int sample = 0; sample < batch_size_; sample++) {
                state_re_[sample + i * batch_size_] = 0;
                state_im_[sample + i * batch_size_] = 0;
            }
        }

#pragma omp simd
        for (int sample = 0; sample < batch_size_; sample++) {
            state_re_[sample] = 1;
            state_im_[sample] = 0;
        }
    }

    void act_single_qubit_gate(UINT target, double matrix_re[2][2], double matrix_im[2][2])
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
            }
        }
    }

    void act_two_qubit_gate(UINT control, UINT target, double matrix_re[4][4],
                            double matrix_im[4][4])
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

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_rx_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{0, -std::sin(theta / 2)}, {-std::sin(theta / 2), 0}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_rx_gate(UINT target, const std::vector<double> &theta)
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
                double cos_half = std::cos(theta[sample] / 2);
                double sin_half = std::sin(theta[sample] / 2);

                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = cos_half * tmp0_re + sin_half * tmp1_im;
                state_im_[sample + i0 * batch_size_] = cos_half * tmp0_im - sin_half * tmp1_re;

                state_re_[sample + i1 * batch_size_] = sin_half * tmp0_im + cos_half * tmp1_re;
                state_im_[sample + i1 * batch_size_] = -sin_half * tmp0_re + cos_half * tmp1_im;
            }
        }
    }

    void act_ry_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), -std::sin(theta / 2)},
                                  {std::sin(theta / 2), std::cos(theta / 2)}};
        double matrix_im[2][2] = {{0, 0}, {0, 0}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_ry_gate(UINT target, const std::vector<double> &theta)
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
                double cos_half = std::cos(theta[sample] / 2);
                double sin_half = std::sin(theta[sample] / 2);

                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = cos_half * tmp0_re - sin_half * tmp1_re;
                state_im_[sample + i0 * batch_size_] = cos_half * tmp0_im - sin_half * tmp1_im;

                state_re_[sample + i1 * batch_size_] = sin_half * tmp0_re + cos_half * tmp1_re;
                state_im_[sample + i1 * batch_size_] = sin_half * tmp0_im + cos_half * tmp1_im;
            }
        }
    }

    void act_rz_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{-std::sin(theta / 2), 0}, {0, std::sin(theta / 2)}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_rz_gate(UINT target, const std::vector<double> &theta)
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
                double cos_half = std::cos(theta[sample] / 2);
                double sin_half = std::sin(theta[sample] / 2);

                double tmp0_re = state_re_[sample + i0 * batch_size_];
                double tmp0_im = state_im_[sample + i0 * batch_size_];
                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i0 * batch_size_] = cos_half * tmp0_re + sin_half * tmp0_im;
                state_im_[sample + i0 * batch_size_] = cos_half * tmp0_im - sin_half * tmp0_re;

                state_re_[sample + i1 * batch_size_] = cos_half * tmp1_re - sin_half * tmp1_im;
                state_im_[sample + i1 * batch_size_] = cos_half * tmp1_im + sin_half * tmp1_re;
            }
        }
    }

    void act_p_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{1, 0}, {0, std::cos(theta)}};
        double matrix_im[2][2] = {{0, 0}, {0, std::sin(theta)}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_p_gate(UINT target, const std::vector<double> &theta)
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
                double cos = std::cos(theta[sample] / 2);
                double sin = std::sin(theta[sample] / 2);

                double tmp1_re = state_re_[sample + i1 * batch_size_];
                double tmp1_im = state_im_[sample + i1 * batch_size_];

                state_re_[sample + i1 * batch_size_] = cos * tmp1_re - sin * tmp1_im;
                state_im_[sample + i1 * batch_size_] = cos * tmp1_im + sin * tmp1_re;
            }
        }
    }

    void act_sx_gate(UINT target)
    {
        static double matrix_re[2][2] = {{0.5, 0.5}, {0.5, 0.5}};
        static double matrix_im[2][2] = {{0.5, -0.5}, {-0.5, 0.5}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_sy_gate(UINT target)
    {
        static double matrix_re[2][2] = {{0.5, -0.5}, {0.5, 0.5}};
        static double matrix_im[2][2] = {{0.5, -0.5}, {0.5, 0.5}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_sw_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static double matrix_re[2][2] = {{inv_sqrt2, -0.5}, {0.5, inv_sqrt2}};
        static double matrix_im[2][2] = {{0, -0.5}, {-0.5, 0}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_t_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static double matrix_re[2][2] = {{1, 0}, {0, inv_sqrt2}};
        static double matrix_im[2][2] = {{0, 0}, {0, inv_sqrt2}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_cnot_gate(UINT control, UINT target)
    {
        static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(control, target, matrix_re, matrix_im);
    }

    void act_cz_gate(UINT control, UINT target)
    {
        static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
        static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(control, target, matrix_re, matrix_im);
    }

    void act_iswaplike_gate(UINT control, UINT target, double theta)
    {
        double matrix_re[4][4] = {
            {1, 0, 0, 0}, {0, std::cos(theta), 0, 0}, {0, 0, std::cos(theta), 0}, {0, 0, 0, 1}};
        double matrix_im[4][4] = {
            {0, 0, 0, 0}, {0, 0, -std::sin(theta), 0}, {0, -std::sin(theta), 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(control, target, matrix_re, matrix_im);
    }

    void act_cx_gate_opt(UINT control, UINT target)
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

    void act_cz_gate_opt(UINT control, UINT target)
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

    void act_depolarizing_gate_2q(UINT control, UINT target, double prob)
    {
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
    }

    std::vector<std::complex<double>> observe(const Observable &obs) const
    {
        std::vector<std::complex<double>> res(batch_size_);

        for (std::uint32_t term_id = 0; term_id < obs.terms.size(); term_id++) {
            std::uint32_t bit_flip_mask = obs.terms[term_id].bit_flip_mask;
            std::uint32_t phase_flip_mask = obs.terms[term_id].phase_flip_mask;
            std::complex<double> coef = obs.terms[term_id].coef;

            if (bit_flip_mask == 0) {
                for (std::uint32_t idx = 0; idx < 1ULL << (n_ - 1); idx++) {
                    std::uint32_t idx1 = idx << 1;
                    std::uint32_t idx2 = idx1 | 1;

                    for (std::uint32_t sample = 0; sample < batch_size_; sample++) {
                        double tmp1 =
                            (std::conj(amplitude(sample, idx1)) * amplitude(sample, idx1)).real();
                        if (__builtin_popcount(idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        double tmp2 =
                            (std::conj(amplitude(sample, idx2)) * amplitude(sample, idx2)).real();
                        if (__builtin_popcount(idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res[sample] += coef * (tmp1 + tmp2);
                    }
                }
            } else {
                for (std::uint32_t idx = 0; idx < 1ULL << (n_ - 1); idx++) {
                    std::uint32_t pivot =
                        sizeof(std::uint32_t) * 8 - __builtin_clz(bit_flip_mask) - 1;
                    std::uint32_t global_phase_90rot_count =
                        __builtin_popcount(bit_flip_mask & phase_flip_mask);
                    std::complex<double> global_phase = PHASE_90ROT()[global_phase_90rot_count % 4];
                    std::uint32_t basis_0 = insert_zero_to_basis_index(idx, pivot);
                    std::uint32_t basis_1 = basis_0 ^ bit_flip_mask;

                    for (std::uint32_t sample = 0; sample < batch_size_; sample++) {
                        double tmp =
                            std::real(amplitude(sample, basis_0) *
                                      std::conj(amplitude(sample, basis_1)) * global_phase * 2.);
                        if (__builtin_popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                        res[sample] += coef * tmp;
                    }
                }
            }
        }

        return res;
    }

private:
    std::vector<double> state_re_;
    std::vector<double> state_im_;
    UINT batch_size_;
    UINT n_;

#ifdef __NEC__
    static asl_random_t rng_;
#endif
    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;
};

#ifdef __NEC__
asl_random_t State::Impl::rng_;
#endif

State::State(UINT n, UINT batch_size) : impl_(std::make_shared<Impl>(n, batch_size)) {}

State::~State() {}

void State::initialize() { Impl::initialize(); }

void State::finalize() { Impl::finalize(); }

std::vector<std::complex<double>> State::get_vector(UINT sample) const
{
    return impl_->get_vector(sample);
}

std::complex<double> State::amplitude(UINT sample, UINT basis) const
{
    return impl_->amplitude(sample, basis);
}

double State::re(UINT sample, UINT basis) const { return impl_->re(sample, basis); }

double State::im(UINT sample, UINT basis) const { return impl_->im(sample, basis); }

double State::get_probability(UINT basis) const { return impl_->get_probability(basis); }

double State::get_probability(UINT sample, UINT basis) const
{
    return impl_->get_probability(sample, basis);
}

std::vector<double> State::get_probability_batched(UINT basis) const
{
    return impl_->get_probability_batched(basis);
}

UINT State::dim() const { return impl_->dim(); }

UINT State::batch_size() const { return impl_->batch_size(); }

void State::set_zero_state() { return impl_->set_zero_state(); }

void State::act_x_gate(UINT target) { impl_->act_x_gate_opt(target); }

void State::act_y_gate(UINT target) { impl_->act_x_gate_opt(target); }

void State::act_z_gate(UINT target) { impl_->act_z_gate_opt(target); }

void State::act_h_gate(UINT target) { impl_->act_h_gate(target); }

void State::act_rx_gate(UINT target, double theta) { impl_->act_rx_gate(target, theta); }

void State::act_rx_gate(UINT target, const std::vector<double> &theta)
{
    impl_->act_rx_gate(target, theta);
}

void State::act_ry_gate(UINT target, double theta) { impl_->act_ry_gate(target, theta); }

void State::act_ry_gate(UINT target, const std::vector<double> &theta)
{
    impl_->act_ry_gate(target, theta);
}

void State::act_rz_gate(UINT target, double theta) { impl_->act_rz_gate(target, theta); }

void State::act_rz_gate(UINT target, const std::vector<double> &theta)
{
    impl_->act_rz_gate(target, theta);
}

void State::act_p_gate(UINT target, double theta) { impl_->act_p_gate(target, theta); }

void State::act_p_gate(UINT target, const std::vector<double> &theta)
{
    impl_->act_p_gate(target, theta);
}

void State::act_sx_gate(UINT target) { impl_->act_sx_gate(target); }

void State::act_sy_gate(UINT target) { impl_->act_sy_gate(target); }

void State::act_sw_gate(UINT target) { impl_->act_sw_gate(target); }

void State::act_t_gate(UINT target) { impl_->act_t_gate(target); }

void State::act_cnot_gate(UINT control, UINT target) { impl_->act_cx_gate_opt(control, target); }

void State::act_iswaplike_gate(UINT control, UINT target, double theta)
{
    impl_->act_iswaplike_gate(control, target, theta);
}

void State::act_cx_gate(UINT control, UINT target) { impl_->act_cx_gate_opt(control, target); }

void State::act_cz_gate(UINT control, UINT target) { impl_->act_cz_gate_opt(control, target); }

void State::act_depolarizing_gate_1q(UINT target, double prob)
{
    impl_->act_depolarizing_gate_1q(target, prob);
}

void State::act_depolarizing_gate_2q(UINT control, UINT target, double prob)
{
    impl_->act_depolarizing_gate_2q(control, target, prob);
}

std::vector<std::complex<double>> State::observe(const Observable &obs) const
{
    return impl_->observe(obs);
}

void State::synchronize() {}

void initialize() { State::initialize(); }

void finalize() { State::finalize(); }

} // namespace veqsim
