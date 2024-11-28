#include <cmath>
#include <random>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <custatevec.h>

#include "state.hpp"

#define HANDLE_ERROR(x)                                                                            \
    {                                                                                              \
        const auto err = x;                                                                        \
        if (err != CUSTATEVEC_STATUS_SUCCESS) {                                                    \
            printf("cuStateVec error \"%s\" at %s:%d\n", custatevecGetErrorString(err), __FILE__,  \
                   __LINE__);                                                                      \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    };

#define HANDLE_CUDA_ERROR(x)                                                                       \
    {                                                                                              \
        const auto err = x;                                                                        \
        if (err != cudaSuccess) {                                                                  \
            printf("CUDA Error: \"%s\" at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);  \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    };

class State::Impl
{
public:
    custatevecHandle_t handle_;
    cuDoubleComplex *state_;
    UINT batch_size_;
    UINT n_;

    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;

    Impl(UINT n, UINT batch_size)
        : n_(n), batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&state_, batch_size * (1ULL << n) * sizeof(cuDoubleComplex)));
        HANDLE_ERROR(custatevecCreate(&handle_));
    }

    ~Impl()
    {
        HANDLE_CUDA_ERROR(cudaFree(state_));
        HANDLE_ERROR(custatevecDestroy(handle_));
    }

    double re(UINT sample, UINT i)
    {
        cuDoubleComplex c;
        HANDLE_CUDA_ERROR(cudaMemcpy(&c, state_ + (1ULL << n_) * sample + i,
                                     sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        return cuCreal(c);
    }

    double im(UINT sample, UINT i)
    {
        cuDoubleComplex c;
        HANDLE_CUDA_ERROR(cudaMemcpy(&c, state_ + (1ULL << n_) * sample + i,
                                     sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        return cuCimag(c);
    }

    double get_probability(UINT i)
    {
        // TODO
    }

    void set_zero_state()
    {
        std::vector<cuDoubleComplex> state(batch_size_ * (1ULL << n_));

        for (ITYPE i = 0; i < state.size(); i += (1ULL << n_)) {
            state[i] = make_cuDoubleComplex(1, 0);
        }

        HANDLE_CUDA_ERROR(cudaMemcpy(state_, state.data(), state.size() * sizeof(cuDoubleComplex),
                                     cudaMemcpyHostToDevice));
    }

    void act_single_qubit_gate(cuDoubleComplex matrix[2][2], UINT target)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};

        // size_t workspace_size;
        // HANDLE_ERROR(custatevecApplyMatrixBatchedGetWorkspaceSize(
        //     handle_, CUDA_C_64F, n_, batch_size_, (1ULL << n_),
        //     CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix, CUDA_C_64F,
        //     CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 1, 0, CUSTATEVEC_COMPUTE_64F,
        //     &workspace_size));

        HANDLE_ERROR(
            custatevecApplyMatrixBatched(handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
                                         CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix,
                                         CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, targets, 1,
                                         nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_two_qubit_gate(cuDoubleComplex matrix[4][4], UINT target, UINT control)
    {
        int32_t targets[] = {static_cast<int32_t>(target), static_cast<int32_t>(control)};

        // size_t workspace_size;
        // HANDLE_ERROR(custatevecApplyMatrixBatchedGetWorkspaceSize(
        //     handle_, CUDA_C_64F, n_, batch_size_, (1ULL << n_),
        //     CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix, CUDA_C_64F,
        //     CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 2, 0, CUSTATEVEC_COMPUTE_64F,
        //     &workspace_size));

        HANDLE_ERROR(
            custatevecApplyMatrixBatched(handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
                                         CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix,
                                         CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, targets, 2,
                                         nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_x_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0)},
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_y_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1)},
            {make_cuDoubleComplex(0, -1), make_cuDoubleComplex(0, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_z_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_h_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(inv_sqrt2, 0)},
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(-inv_sqrt2, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_rx_gate(double theta, UINT target)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, 0), make_cuDoubleComplex(0, -sin_half)},
            {make_cuDoubleComplex(0, -sin_half), make_cuDoubleComplex(cos_half, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_ry_gate(double theta, UINT target)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, 0), make_cuDoubleComplex(-sin_half, 0)},
            {make_cuDoubleComplex(sin_half, 0), make_cuDoubleComplex(cos_half, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_rz_gate(double theta, UINT target)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, -sin_half), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(cos_half, sin_half)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_sx_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(0.5, -0.5)},
            {make_cuDoubleComplex(0.5, -0.5), make_cuDoubleComplex(0.5, 0.5)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_sy_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(-0.5, -0.5)},
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(0.5, 0.5)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_sw_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(-0.5, -0.5)},
            {make_cuDoubleComplex(0.5, -0.5), make_cuDoubleComplex(inv_sqrt2, 0)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_t_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(inv_sqrt2, inv_sqrt2)}};

        act_single_qubit_gate(matrix, target);
    }

    void act_cnot_gate(UINT target, UINT control)
    {
        static cuDoubleComplex matrix[4][4] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(1, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
             make_cuDoubleComplex(0, 0)},
        };

        act_two_qubit_gate(matrix, target, control);
    }

    void act_cx_gate(UINT target, UINT control)
    {
        static cuDoubleComplex matrix[4][4] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(1, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
        };

        act_two_qubit_gate(matrix, target, control);
    }

    void act_cz_gate(UINT target, UINT control)
    {
        static cuDoubleComplex matrix[4][4] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(-1, 0)},
        };

        act_two_qubit_gate(matrix, target, control);
    }

    void act_iswaplike_gate(double theta, UINT target, UINT control)
    {
        static cuDoubleComplex matrix[4][4] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(std::cos(theta), 0),
             make_cuDoubleComplex(0, -std::sin(theta)), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -std::sin(theta)),
             make_cuDoubleComplex(std::cos(theta), 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
             make_cuDoubleComplex(1, 0)},
        };

        act_two_qubit_gate(matrix, target, control);
    }

    void act_depolarizing_gate_1q(UINT target, double prob)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};
        std::vector<int> matrix_indices(batch_size_);

        cuDoubleComplex matrices[] = {// I gate
                                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
                                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
                                      // X gate
                                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
                                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
                                      // Y gate
                                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1),
                                      make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 0),
                                      // Z gate
                                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0),
                                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0)};

        for (int sample = 0; sample < batch_size_; sample++) {
            double dice = dist_(mt_engine_);

            if (dice < prob / 3.0) {
                matrix_indices[sample] = 1;
            } else if (dice < prob * 2.0 / 3.0) {
                matrix_indices[sample] = 2;
            } else if (dice < prob) {
                matrix_indices[sample] = 3;
            }
        }

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrix_indices.data(), matrices, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 4, targets, 1, nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_depolarizing_gate_2q(UINT target, UINT control, double prob)
    {
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
    }

    void synchronize() { HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); }
};

State::State(UINT n, UINT batch_size) : impl_(std::make_shared<Impl>(n, batch_size)) {}

State::~State() {}

double State::re(UINT sample, UINT i) { return impl_->re(sample, i); }

double State::im(UINT sample, UINT i) { return impl_->im(sample, i); }

double State::get_probability(UINT i) { return impl_->get_probability(i); }

void State::set_zero_state() { return impl_->set_zero_state(); }

void State::act_x_gate(UINT target) { impl_->act_x_gate(target); }

void State::act_y_gate(UINT target) { impl_->act_x_gate(target); }

void State::act_z_gate(UINT target) { impl_->act_z_gate(target); }

void State::act_h_gate(UINT target) { impl_->act_h_gate(target); }

void State::act_rx_gate(double theta, UINT target) { impl_->act_rx_gate(theta, target); }

void State::act_ry_gate(double theta, UINT target) { impl_->act_ry_gate(theta, target); }

void State::act_rz_gate(double theta, UINT target) { impl_->act_rz_gate(theta, target); }

void State::act_sx_gate(UINT target) { impl_->act_sx_gate(target); }

void State::act_sy_gate(UINT target) { impl_->act_sy_gate(target); }

void State::act_sw_gate(UINT target) { impl_->act_sw_gate(target); }

void State::act_t_gate(UINT target) { impl_->act_t_gate(target); }

void State::act_cnot_gate(UINT target, UINT control) { impl_->act_cnot_gate(target, control); }

void State::act_iswaplike_gate(double theta, UINT target, UINT control)
{

    impl_->act_iswaplike_gate(theta, target, control);
}

void State::act_cx_gate(UINT target, UINT control) { impl_->act_cx_gate(target, control); }

void State::act_cz_gate(UINT target, UINT control) { impl_->act_cz_gate(target, control); }

void State::act_depolarizing_gate_1q(UINT target, double prob)
{
    impl_->act_depolarizing_gate_1q(target, prob);
}

void State::act_depolarizing_gate_2q(UINT target, UINT control, double prob)
{
    impl_->act_depolarizing_gate_2q(target, control, prob);
}

void State::synchronize() { impl_->synchronize(); }
