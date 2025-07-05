// vim: set ft=cuda:
#include <cmath>
#include <random>
#include <stdexcept>
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

namespace veqsim
{

class State::Impl
{
public:
    Impl(UINT n, UINT batch_size)
        : n_(n), batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&state_, batch_size * (1ULL << n) * sizeof(cuDoubleComplex)));
    }

    ~Impl() { HANDLE_CUDA_ERROR(cudaFree(state_)); }

    static void initialize() { HANDLE_ERROR(custatevecCreate(&handle_)); }

    static void finalize() { HANDLE_ERROR(custatevecDestroy(handle_)); }

    std::vector<std::complex<double>> get_vector(UINT sample) const
    {
        std::vector<std::complex<double>> sv(1ULL << n_);

        HANDLE_CUDA_ERROR(cudaMemcpy(sv.data(), state_ + (1ULL << n_) * sample,
                                     (1ULL << n_) * sizeof(cuDoubleComplex),
                                     cudaMemcpyDeviceToHost));

        return sv;
    }

    std::complex<double> amplitude(UINT sample, UINT i)
    {
        cuDoubleComplex c;
        HANDLE_CUDA_ERROR(cudaMemcpy(&c, state_ + (1ULL << n_) * sample + i,
                                     sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        return std::complex(cuCreal(c), cuCimag(c));
    }

    double re(UINT sample, UINT i) { return amplitude(sample, i).real(); }

    double im(UINT sample, UINT i) { return amplitude(sample, i).imag(); }

    double get_probability(UINT i) { throw std::runtime_error("Not implemented"); }

    double get_probability(UINT sample, UINT i) { return std::norm(amplitude(sample, i)); }

    std::vector<double> get_probability_batched(UINT i) const
    {
        std::vector<double> probs(batch_size_);
        std::vector<custatevecIndex_t> mask_bit_string(batch_size_);
        std::vector<int32_t> mask_ordering(n_);
        std::iota(mask_ordering.begin(), mask_ordering.end(), 0);

        for (int i = 0; i < n_; i++) {
            mask_ordering[i] = i;
        }

        HANDLE_ERROR(custatevecAbs2SumArrayBatched(
            handle_,                // custatevecHandle_t handle
            state_,                 // const void *batchedSv
            CUDA_C_64F,             // cudaDataType_t svDataType
            n_,                     // const uint32_t nIndexBits
            batch_size_,            // const uint32_t nSVs
            1ULL << n_,             // const custatevecIndex_t svStride
            probs.data(),           // double *abs2sumArrays
            1,                      // const custatevecIndex_t abs2sumArrayStride
            nullptr,                // const int32_t *bitOrdering
            0,                      // const uint32_t bitOrderingLen
            mask_bit_string.data(), // const custatevecIndex_t *maskBitStrings
            mask_ordering.data(),   // const int32_t *maskOrdering
            n_                      // const uint32_t maskLen
            ));

        return probs;
    }

    UINT dim() const { return 1ULL << n_; }

    UINT batch_size() const { return batch_size_; }

    void set_zero_state()
    {
        std::vector<cuDoubleComplex> state(batch_size_ * (1ULL << n_));

        for (ITYPE i = 0; i < state.size(); i += (1ULL << n_)) {
            state[i] = make_cuDoubleComplex(1, 0);
        }

        HANDLE_CUDA_ERROR(cudaMemcpy(state_, state.data(), state.size() * sizeof(cuDoubleComplex),
                                     cudaMemcpyHostToDevice));
    }

    void act_single_qubit_gate(UINT target, cuDoubleComplex matrix[2][2])
    {
        int32_t targets[] = {static_cast<int32_t>(target)};

        // size_t workspace_size;
        // HANDLE_ERROR(custatevecApplyMatrixBatchedGetWorkspaceSize(
        //     handle_, CUDA_C_64F, n_, batch_size_, (1ULL << n_),
        //     CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix, CUDA_C_64F,
        //     CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 1, 0, CUSTATEVEC_COMPUTE_64F,
        //     &workspace_size));

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_,                              // custatevecHandle_t handle
            state_,                               // void *batchedSv
            CUDA_C_64F,                           // cudaDataType_t svDataType
            n_,                                   // const uint32_t nIndexBits
            batch_size_,                          // const uint32_t nSVs
            1ULL << n_,                           // custatevecIndex_t svStride
            CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, // custatevecMatrixMapType_t mapType
            nullptr,                              // const int32_t *matrixIndices
            matrix,                               // const void *matrices
            CUDA_C_64F,                           // cudaDataType_t matrixDataType
            CUSTATEVEC_MATRIX_LAYOUT_ROW,         // custatevecMatrixLayout_t layout
            0,                                    // const int32_t adjoint
            1,                                    // const uint32_t nMatrices
            targets,                              // const int32_t *targets
            1,                                    // const uint32_t nTargets
            nullptr,                              // const int32_t *controls
            nullptr,                              // const int32_t *controlBitValues
            0,                                    // const uint32_t nControls
            CUSTATEVEC_COMPUTE_64F,               // custatevecComputeType_t computeType
            nullptr,                              // void *extraWorkspace
            0                                     // size_t extraWorkspaceSizeInBytes
            ));
    }

    void act_two_qubit_gate(UINT control, UINT target, cuDoubleComplex matrix[4][4])
    {
        int32_t targets[] = {static_cast<int32_t>(target), static_cast<int32_t>(control)};

        // size_t workspace_size;
        // HANDLE_ERROR(custatevecApplyMatrixBatchedGetWorkspaceSize(
        //     handle_, CUDA_C_64F, n_, batch_size_, (1ULL << n_),
        //     CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix, CUDA_C_64F,
        //     CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, 1, 2, 0, CUSTATEVEC_COMPUTE_64F,
        //     &workspace_size));

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_,                              // custatevecHandle_t handle
            state_,                               // void *batchedSv
            CUDA_C_64F,                           // cudaDataType_t svDataType
            n_,                                   // const uint32_t nIndexBits
            batch_size_,                          // const uint32_t nSVs
            1ULL << n_,                           // custatevecIndex_t svStride
            CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, // custatevecMatrixMapType_t mapType
            nullptr,                              // const int32_t *matrixIndices
            matrix,                               // const void *matrices
            CUDA_C_64F,                           // cudaDataType_t matrixDataType
            CUSTATEVEC_MATRIX_LAYOUT_ROW,         // custatevecMatrixLayout_t layout
            0,                                    // const int32_t adjoint
            1,                                    // const uint32_t nMatrices
            targets,                              // const int32_t *targets
            2,                                    // const uint32_t nTargets
            nullptr,                              // const int32_t *controls
            nullptr,                              // const int32_t *controlBitValues
            0,                                    // const uint32_t nControls
            CUSTATEVEC_COMPUTE_64F,               // custatevecComputeType_t computeType
            nullptr,                              // void *extraWorkspace
            0                                     // size_t extraWorkspaceSizeInBytes
            ));
    }

    void act_x_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0)},
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_y_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1)},
            {make_cuDoubleComplex(0, -1), make_cuDoubleComplex(0, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_z_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_h_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(inv_sqrt2, 0)},
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(-inv_sqrt2, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_rx_gate(UINT target, double theta)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, 0), make_cuDoubleComplex(0, -sin_half)},
            {make_cuDoubleComplex(0, -sin_half), make_cuDoubleComplex(cos_half, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_rx_gate(UINT target, const std::vector<double> &theta)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};
        std::vector<int> matrix_indices(batch_size_);
        std::vector<cuDoubleComplex> matrices(batch_size_ * 4);

        for (int i = 0; i < batch_size_; i++) {
            double cos_half = std::cos(theta[i] / 2), sin_half = std::sin(theta[i] / 2);

            matrix_indices[i] = i;
            matrices[i * 4 + 0] = make_cuDoubleComplex(cos_half, 0);
            matrices[i * 4 + 1] = make_cuDoubleComplex(0, -sin_half);
            matrices[i * 4 + 2] = make_cuDoubleComplex(0, -sin_half);
            matrices[i * 4 + 3] = make_cuDoubleComplex(cos_half, 0);
        }

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrix_indices.data(), matrices.data(),
            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, batch_size_, targets, 1, nullptr, nullptr,
            0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_ry_gate(UINT target, double theta)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, 0), make_cuDoubleComplex(-sin_half, 0)},
            {make_cuDoubleComplex(sin_half, 0), make_cuDoubleComplex(cos_half, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_ry_gate(UINT target, const std::vector<double> &theta)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};
        std::vector<int> matrix_indices(batch_size_);
        std::vector<cuDoubleComplex> matrices(batch_size_ * 4);

        for (int i = 0; i < batch_size_; i++) {
            double cos_half = std::cos(theta[i] / 2), sin_half = std::sin(theta[i] / 2);

            matrix_indices[i] = i;
            matrices[i * 4 + 0] = make_cuDoubleComplex(cos_half, 0);
            matrices[i * 4 + 1] = make_cuDoubleComplex(-sin_half, 0);
            matrices[i * 4 + 2] = make_cuDoubleComplex(sin_half, 0);
            matrices[i * 4 + 3] = make_cuDoubleComplex(cos_half, 0);
        }

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrix_indices.data(), matrices.data(),
            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, batch_size_, targets, 1, nullptr, nullptr,
            0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_rz_gate(UINT target, double theta)
    {
        double cos_half = std::cos(theta / 2), sin_half = std::sin(theta / 2);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(cos_half, -sin_half), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(cos_half, sin_half)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_rz_gate(UINT target, const std::vector<double> &theta)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};
        std::vector<int> matrix_indices(batch_size_);
        std::vector<cuDoubleComplex> matrices(batch_size_ * 4);

        for (int i = 0; i < batch_size_; i++) {
            double cos_half = std::cos(theta[i] / 2), sin_half = std::sin(theta[i] / 2);

            matrix_indices[i] = i;
            matrices[i * 4 + 0] = make_cuDoubleComplex(cos_half, -sin_half);
            matrices[i * 4 + 1] = make_cuDoubleComplex(0, 0);
            matrices[i * 4 + 2] = make_cuDoubleComplex(0, 0);
            matrices[i * 4 + 3] = make_cuDoubleComplex(cos_half, sin_half);
        }

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrix_indices.data(), matrices.data(),
            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, batch_size_, targets, 1, nullptr, nullptr,
            0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_p_gate(UINT target, double theta)
    {
        double cos = std::cos(theta), sin = std::sin(theta);
        cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(cos, sin)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_p_gate(UINT target, const std::vector<double> &theta)
    {
        int32_t targets[] = {static_cast<int32_t>(target)};
        std::vector<int> matrix_indices(batch_size_);
        std::vector<cuDoubleComplex> matrices(batch_size_ * 4);

        for (int i = 0; i < batch_size_; i++) {
            double cos = std::cos(theta[i] / 2), sin = std::sin(theta[i] / 2);

            matrix_indices[i] = i;
            matrices[i * 4 + 0] = make_cuDoubleComplex(1, 0);
            matrices[i * 4 + 1] = make_cuDoubleComplex(0, 0);
            matrices[i * 4 + 2] = make_cuDoubleComplex(0, 0);
            matrices[i * 4 + 3] = make_cuDoubleComplex(cos, sin);
        }

        HANDLE_ERROR(custatevecApplyMatrixBatched(
            handle_, state_, CUDA_C_64F, n_, batch_size_, 1ULL << n_,
            CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrix_indices.data(), matrices.data(),
            CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, batch_size_, targets, 1, nullptr, nullptr,
            0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
    }

    void act_sx_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(0.5, -0.5)},
            {make_cuDoubleComplex(0.5, -0.5), make_cuDoubleComplex(0.5, 0.5)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_sy_gate(UINT target)
    {
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(-0.5, -0.5)},
            {make_cuDoubleComplex(0.5, 0.5), make_cuDoubleComplex(0.5, 0.5)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_sw_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(inv_sqrt2, 0), make_cuDoubleComplex(-0.5, -0.5)},
            {make_cuDoubleComplex(0.5, -0.5), make_cuDoubleComplex(inv_sqrt2, 0)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_t_gate(UINT target)
    {
        static double inv_sqrt2 = 1 / std::sqrt(2);
        static cuDoubleComplex matrix[2][2] = {
            {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)},
            {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(inv_sqrt2, inv_sqrt2)}};

        act_single_qubit_gate(target, matrix);
    }

    void act_cx_gate(UINT control, UINT target)
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

        act_two_qubit_gate(control, target, matrix);
    }

    void act_cz_gate(UINT control, UINT target)
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

        act_two_qubit_gate(control, target, matrix);
    }

    void act_iswaplike_gate(UINT control, UINT target, double theta)
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

        act_two_qubit_gate(control, target, matrix);
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

    void act_depolarizing_gate_2q(UINT control, UINT target, double prob)
    {
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
    }

    std::vector<std::complex<double>> observe(const Observable &obs) const
    {
        throw std::runtime_error("Not implemented");
    }

    void synchronize() { HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); }

private:
    static custatevecHandle_t handle_;
    cuDoubleComplex *state_;
    UINT batch_size_;
    UINT n_;

    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;
};

custatevecHandle_t State::Impl::handle_;

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

void State::act_x_gate(UINT target) { impl_->act_x_gate(target); }

void State::act_y_gate(UINT target) { impl_->act_x_gate(target); }

void State::act_z_gate(UINT target) { impl_->act_z_gate(target); }

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

void State::act_cnot_gate(UINT control, UINT target) { impl_->act_cx_gate(control, target); }

void State::act_iswaplike_gate(UINT control, UINT target, double theta)
{

    impl_->act_iswaplike_gate(control, target, theta);
}

void State::act_cx_gate(UINT control, UINT target) { impl_->act_cx_gate(control, target); }

void State::act_cz_gate(UINT control, UINT target) { impl_->act_cz_gate(control, target); }

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

void State::synchronize() { impl_->synchronize(); }

void initialize() { State::initialize(); }

void finalize() { State::finalize(); }

} // namespace veqsim
