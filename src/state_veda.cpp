#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <veda.h>

#include "state.hpp"

#define VEDA(err) check(err, __FILE__, __LINE__)

void check(VEDAresult err, const char* file, const int line) {
    if(err != VEDA_SUCCESS) {
        const char *name, *str;
        vedaGetErrorName(err, &name);
        vedaGetErrorString(err, &str);
        printf("%s: %s @ %s:%i\n", name, str, file, line);
        exit(1);
    }
}

class State::Impl
{
public:
    UINT batch_size_;
    UINT n_;

    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;

    VEDAcontext context_;
    VEDAmodule module_;
    VEDAdeviceptr state_re_ptr_, state_im_ptr_;

    VEDAfunction get_probability_;
    VEDAfunction set_zero_state_;
    VEDAfunction act_single_qubit_gate_;
    VEDAfunction act_two_qubit_gate_;
    VEDAfunction act_x_gate_opt_;
    VEDAfunction act_y_gate_opt_;
    VEDAfunction act_z_gate_opt_;
    VEDAfunction act_cnot_gate_opt_;
    VEDAfunction act_cx_gate_opt_;
    VEDAfunction act_cz_gate_opt_;
    VEDAfunction act_depolarizing_gate_1q_;

    Impl(UINT n, UINT batch_size)
        : n_(n), batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
        VEDA(vedaInit(0));
        VEDA(vedaDevicePrimaryCtxRetain(&context_, 0));
        VEDA(vedaCtxPushCurrent(context_));
        VEDA(vedaModuleLoad(&module_, "./libqsim_device.vso"));

        VEDA(vedaModuleGetFunction(&get_probability_, module_, "get_probability"));
        VEDA(vedaModuleGetFunction(&set_zero_state_, module_, "set_zero_state"));
        VEDA(vedaModuleGetFunction(&act_single_qubit_gate_, module_, "act_single_qubit_gate"));
        VEDA(vedaModuleGetFunction(&act_two_qubit_gate_, module_, "act_two_qubit_gate"));
        VEDA(vedaModuleGetFunction(&act_x_gate_opt_, module_, "act_x_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_y_gate_opt_, module_, "act_y_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_z_gate_opt_, module_, "act_z_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_cnot_gate_opt_, module_, "act_cnot_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_cx_gate_opt_, module_, "act_cx_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_cz_gate_opt_, module_, "act_cz_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_depolarizing_gate_1q_, module_, "act_depolarizing_gate_1q"));

        VEDA(vedaMemAlloc(&state_re_ptr_, (1ULL << n) * batch_size * sizeof(double)));
        VEDA(vedaMemAlloc(&state_im_ptr_, (1ULL << n) * batch_size * sizeof(double)));
    }

    ~Impl()
    {
        VEDA(vedaMemFree(state_re_ptr_));
        VEDA(vedaMemFree(state_im_ptr_));
        VEDA(vedaExit());
    }

    std::complex<double> amplitude(UINT sample, UINT i)
    {
        double re = 0.0, im = 0.0;

        VEDA(vedaMemcpyDtoH(&re, state_re_ptr_ + (sample + i * batch_size_) * sizeof(double),
                            sizeof(double)));
        VEDA(vedaMemcpyDtoH(&im, state_im_ptr_ + (sample + i * batch_size_) * sizeof(double),
                            sizeof(double)));

        return std::complex(re, im);
    }

    double re(UINT sample, UINT i) {
        double re = 0.0;

        VEDA(vedaMemcpyDtoH(&re, state_re_ptr_ + (sample + i * batch_size_) * sizeof(double),
                            sizeof(double)));

        return re;
    }

    double im(UINT sample, UINT i) {
        double im = 0.0;

        VEDA(vedaMemcpyDtoH(&im, state_im_ptr_ + (sample + i * batch_size_) * sizeof(double),
                            sizeof(double)));

        return im;
    }

    double get_probability(UINT i)
    {
        double prob = 0.0;

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetStack(args, 2, &prob, VEDA_ARGS_INTENT_OUT, sizeof(double)));
        VEDA(vedaArgsSetU64(args, 3, batch_size_));
        VEDA(vedaArgsSetU64(args, 4, n_));
        VEDA(vedaLaunchKernelEx(get_probability_, 0, args, 1, nullptr));
        VEDA(vedaArgsDestroy(args));

        return prob;
    }

    UINT dim() const { return 1ULL << n_; }

    UINT batch_size() const { return batch_size_; }

    void set_zero_state()
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, batch_size_));
        VEDA(vedaArgsSetU64(args, 3, n_));
        VEDA(vedaLaunchKernel(set_zero_state_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_single_qubit_gate(double matrix_re[2][2], double matrix_im[2][2], UINT target)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetStack(args, 2, matrix_re, VEDA_ARGS_INTENT_IN, 4 * sizeof(double)));
        VEDA(vedaArgsSetStack(args, 3, matrix_im, VEDA_ARGS_INTENT_IN, 4 * sizeof(double)));
        VEDA(vedaArgsSetU64(args, 4, target));
        VEDA(vedaArgsSetU64(args, 5, batch_size_));
        VEDA(vedaArgsSetU64(args, 6, n_));
        VEDA(vedaLaunchKernel(act_single_qubit_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_two_qubit_gate(double matrix_re[4][4], double matrix_im[4][4], UINT target,
                            UINT control)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetStack(args, 2, matrix_re, VEDA_ARGS_INTENT_IN, 16 * sizeof(double)));
        VEDA(vedaArgsSetStack(args, 3, matrix_im, VEDA_ARGS_INTENT_IN, 16 * sizeof(double)));
        VEDA(vedaArgsSetU64(args, 4, target));
        VEDA(vedaArgsSetU64(args, 5, control));
        VEDA(vedaArgsSetU64(args, 6, batch_size_));
        VEDA(vedaArgsSetU64(args, 7, n_));
        VEDA(vedaLaunchKernel(act_two_qubit_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_x_gate_opt(UINT target)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, batch_size_));
        VEDA(vedaArgsSetU64(args, 4, n_));
        VEDA(vedaLaunchKernel(act_x_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_y_gate_opt(UINT target)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, batch_size_));
        VEDA(vedaArgsSetU64(args, 4, n_));
        VEDA(vedaLaunchKernel(act_y_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_z_gate_opt(UINT target)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, batch_size_));
        VEDA(vedaArgsSetU64(args, 4, n_));
        VEDA(vedaLaunchKernel(act_z_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
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
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, control));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_cnot_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
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

    void act_cx_gate_opt(UINT target, UINT control)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, control));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_cx_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_cz_gate_opt(UINT target, UINT control)
    {
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 2, target));
        VEDA(vedaArgsSetU64(args, 3, control));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_cz_gate_opt_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_depolarizing_gate_1q(UINT target, double prob)
    {
        std::vector<double> dice(batch_size_);
        std::vector<int> x_samples, y_samples, z_samples;

        for (int sample = 0; sample < batch_size_; sample++) {
            dice[sample] = dist_(mt_engine_);
        }

        for (int sample = 0; sample < batch_size_; sample++) {
            if (dice[sample] < prob / 3.0) {
                x_samples.push_back(sample);
            } else if (dice[sample] < prob * 2.0 / 3.0) {
                y_samples.push_back(sample);
            } else if (dice[sample] < prob) {
                z_samples.push_back(sample);
            }
        }

        VEDAdeviceptr x_samples_ptr_, y_samples_ptr_, z_samples_ptr_;
        VEDA(vedaMemAlloc(&x_samples_ptr_, x_samples.size() * sizeof(int)));
        VEDA(vedaMemAlloc(&y_samples_ptr_, y_samples.size() * sizeof(int)));
        VEDA(vedaMemAlloc(&z_samples_ptr_, z_samples.size() * sizeof(int)));
        VEDA(vedaMemcpyHtoD(x_samples_ptr_, x_samples.data(), x_samples.size() * sizeof(int)));
        VEDA(vedaMemcpyHtoD(y_samples_ptr_, y_samples.data(), y_samples.size() * sizeof(int)));
        VEDA(vedaMemcpyHtoD(z_samples_ptr_, z_samples.data(), z_samples.size() * sizeof(int)));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, x_samples_ptr_));
        VEDA(vedaArgsSetVPtr(args, 3, y_samples_ptr_));
        VEDA(vedaArgsSetVPtr(args, 4, z_samples_ptr_));
        VEDA(vedaArgsSetU64(args, 5, x_samples.size()));
        VEDA(vedaArgsSetU64(args, 6, y_samples.size()));
        VEDA(vedaArgsSetU64(args, 7, z_samples.size()));
        VEDA(vedaArgsSetU64(args, 8, target));
        VEDA(vedaArgsSetU64(args, 9, batch_size_));
        VEDA(vedaArgsSetU64(args, 10, n_));
        VEDA(vedaLaunchKernel(act_depolarizing_gate_1q_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemFree(x_samples_ptr_));
        VEDA(vedaMemFree(y_samples_ptr_));
        VEDA(vedaMemFree(z_samples_ptr_));
    }

    void act_depolarizing_gate_2q(UINT target, UINT control, double prob)
    {
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
    }
};

State::State(UINT n, UINT batch_size) : impl_(std::make_shared<Impl>(n, batch_size)) {}

State::~State() {}

std::complex<double> State::amplitude(UINT sample, UINT i) const
{
    return impl_->amplitude(sample, i);
}

double State::re(UINT sample, UINT i) const { return impl_->re(sample, i); }

double State::im(UINT sample, UINT i) const { return impl_->im(sample, i); }

double State::get_probability(UINT i) const { return impl_->get_probability(i); }

UINT State::dim() const { return impl_->dim(); }

UINT State::batch_size() const { return impl_->batch_size(); }

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

void State::act_cx_gate(UINT target, UINT control) { impl_->act_cx_gate_opt(target, control); }

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
