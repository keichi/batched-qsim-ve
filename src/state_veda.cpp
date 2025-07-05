#include <cmath>
#include <random>

#include <veda.h>

#include "observable.hpp"
#include "state.hpp"

#define VEDA(err) check(err, __FILE__, __LINE__)

void check(VEDAresult err, const char *file, const int line)
{
    if (err != VEDA_SUCCESS) {
        const char *name, *str;
        vedaGetErrorName(err, &name);
        vedaGetErrorString(err, &str);
        printf("%s: %s @ %s:%i\n", name, str, file, line);
        exit(1);
    }
}

namespace veqsim
{

class State::Impl
{
public:
    Impl(UINT n, UINT batch_size)
        : n_(n), batch_size_(batch_size), mt_engine_(seed_gen_()), dist_(0.0, 1.0)
    {
        VEDA(vedaMemAlloc(&state_re_ptr_, (1ULL << n) * batch_size * sizeof(double)));
        VEDA(vedaMemAlloc(&state_im_ptr_, (1ULL << n) * batch_size * sizeof(double)));
        VEDA(vedaMemAlloc(&dice_ptr_, batch_size_ * sizeof(double)));
        VEDA(vedaMemAlloc(&x_samples_ptr_, batch_size_ * sizeof(int)));
        VEDA(vedaMemAlloc(&y_samples_ptr_, batch_size_ * sizeof(int)));
        VEDA(vedaMemAlloc(&z_samples_ptr_, batch_size_ * sizeof(int)));
    }

    ~Impl()
    {
        VEDA(vedaMemFree(state_re_ptr_));
        VEDA(vedaMemFree(state_im_ptr_));
        VEDA(vedaMemFree(dice_ptr_));
        VEDA(vedaMemFree(x_samples_ptr_));
        VEDA(vedaMemFree(y_samples_ptr_));
        VEDA(vedaMemFree(z_samples_ptr_));
    }

    static void initialize()
    {
        VEDA(vedaInit(0));
        VEDA(vedaDevicePrimaryCtxRetain(&context_, 0));
        VEDA(vedaCtxPushCurrent(context_));
        VEDA(vedaModuleLoad(&mod_, "libveqsim-device.vso"));

        VEDA(vedaModuleGetFunction(&get_vector_, mod_, "get_vector"));
        VEDA(vedaModuleGetFunction(&get_probability_average_, mod_, "get_probability_average"));
        VEDA(vedaModuleGetFunction(&get_probability_batched_, mod_, "get_probability_batched"));
        VEDA(vedaModuleGetFunction(&set_zero_state_, mod_, "set_zero_state"));
        VEDA(vedaModuleGetFunction(&act_single_qubit_gate_, mod_, "act_single_qubit_gate"));
        VEDA(vedaModuleGetFunction(&act_two_qubit_gate_, mod_, "act_two_qubit_gate"));
        VEDA(vedaModuleGetFunction(&act_rx_gate_, mod_, "act_rx_gate"));
        VEDA(vedaModuleGetFunction(&act_ry_gate_, mod_, "act_ry_gate"));
        VEDA(vedaModuleGetFunction(&act_rz_gate_, mod_, "act_rz_gate"));
        VEDA(vedaModuleGetFunction(&act_p_gate_, mod_, "act_p_gate"));
        VEDA(vedaModuleGetFunction(&act_x_gate_opt_, mod_, "act_x_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_y_gate_opt_, mod_, "act_y_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_z_gate_opt_, mod_, "act_z_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_cx_gate_opt_, mod_, "act_cx_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_cz_gate_opt_, mod_, "act_cz_gate_opt"));
        VEDA(vedaModuleGetFunction(&act_depolarizing_gate_1q_, mod_, "act_depolarizing_gate_1q"));
        VEDA(vedaModuleGetFunction(&observe_, mod_, "observe"));
    }

    static void finalize()
    {
        VEDA(vedaCtxSynchronize());
        VEDA(vedaCtxDestroy(context_));

        VEDA(vedaExit());
    }

    std::vector<std::complex<double>> get_vector(UINT sample) const
    {
        VEDAdeviceptr sv_ptr;
        std::vector<std::complex<double>> sv(1ULL << n_);

        VEDA(vedaMemAllocAsync(&sv_ptr, (1ULL << n_) * sizeof(std::complex<double>), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, sv_ptr));
        VEDA(vedaArgsSetU64(args, 3, sample));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(get_vector_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemcpyDtoHAsync(sv.data(), sv_ptr, (1ULL << n_) * sizeof(std::complex<double>), 0));
        VEDA(vedaMemFreeAsync(sv_ptr, 0));
        VEDA(vedaStreamSynchronize(0));

        return sv;
    }

    std::complex<double> amplitude(UINT sample, UINT i)
    {
        return std::complex(re(sample, i), im(sample, i));
    }

    double re(UINT sample, UINT i)
    {
        double re = 0.0;

        VEDA(vedaMemcpyDtoH(&re, state_re_ptr_ + (sample + i * batch_size_) * sizeof(double),
                            sizeof(double)));

        return re;
    }

    double im(UINT sample, UINT i)
    {
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
        VEDA(vedaArgsSetStack(args, 0, &prob, VEDA_ARGS_INTENT_OUT, sizeof(double)));
        VEDA(vedaArgsSetVPtr(args, 1, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, state_im_ptr_));
        VEDA(vedaArgsSetU64(args, 3, i));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(get_probability_average_, 0, args));
        VEDA(vedaArgsDestroy(args));
        VEDA(vedaCtxSynchronize());

        return prob;
    }

    double get_probability(UINT sample, UINT i)
    {
        return std::norm(amplitude(sample, i));
    }

    std::vector<double> get_probability_batched(UINT i) const
    {
        VEDAdeviceptr prob_ptr;
        std::vector<double> probs(batch_size_);

        VEDA(vedaMemAllocAsync(&prob_ptr, batch_size_ * sizeof(double), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, prob_ptr));
        VEDA(vedaArgsSetU64(args, 3, i));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(get_probability_batched_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemcpyDtoHAsync(probs.data(), prob_ptr, batch_size_ * sizeof(double), 0));
        VEDA(vedaMemFreeAsync(prob_ptr, 0));
        VEDA(vedaStreamSynchronize(0));

        return probs;
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

    void act_single_qubit_gate(UINT target, double matrix_re[2][2], double matrix_im[2][2])
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

    void act_two_qubit_gate(UINT control, UINT target,
                            double matrix_re[4][4], double matrix_im[4][4])
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
        VEDAdeviceptr theta_ptr;
        VEDA(vedaMemAllocAsync(&theta_ptr, batch_size_ * sizeof(double), 0));
        VEDA(vedaMemcpyHtoDAsync(theta_ptr, theta.data(), batch_size_ * sizeof(double), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, theta_ptr));
        VEDA(vedaArgsSetU64(args, 3, target));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_rx_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemFreeAsync(theta_ptr, 0));
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
        VEDAdeviceptr theta_ptr;
        VEDA(vedaMemAllocAsync(&theta_ptr, batch_size_ * sizeof(double), 0));
        VEDA(vedaMemcpyHtoDAsync(theta_ptr, theta.data(), batch_size_ * sizeof(double), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, theta_ptr));
        VEDA(vedaArgsSetU64(args, 3, target));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_ry_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemFreeAsync(theta_ptr, 0));
    }

    void act_rz_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{std::cos(theta / 2), 0}, {0, std::cos(theta / 2)}};
        double matrix_im[2][2] = {{-std::sin(theta / 2), 0}, {0, std::sin(theta / 2)}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_rz_gate(UINT target, const std::vector<double> &theta)
    {
        VEDAdeviceptr theta_ptr;
        VEDA(vedaMemAllocAsync(&theta_ptr, batch_size_ * sizeof(double), 0));
        VEDA(vedaMemcpyHtoDAsync(theta_ptr, theta.data(), batch_size_ * sizeof(double), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, theta_ptr));
        VEDA(vedaArgsSetU64(args, 3, target));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_rz_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemFreeAsync(theta_ptr, 0));
    }

    void act_p_gate(UINT target, double theta)
    {
        double matrix_re[2][2] = {{1, 0}, {0, std::cos(theta)}};
        double matrix_im[2][2] = {{0, 0}, {0, std::sin(theta)}};

        act_single_qubit_gate(target, matrix_re, matrix_im);
    }

    void act_p_gate(UINT target, const std::vector<double> &theta)
    {
        VEDAdeviceptr theta_ptr;
        VEDA(vedaMemAllocAsync(&theta_ptr, batch_size_ * sizeof(double), 0));
        VEDA(vedaMemcpyHtoDAsync(theta_ptr, theta.data(), batch_size_ * sizeof(double), 0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, theta_ptr));
        VEDA(vedaArgsSetU64(args, 3, target));
        VEDA(vedaArgsSetU64(args, 4, batch_size_));
        VEDA(vedaArgsSetU64(args, 5, n_));
        VEDA(vedaLaunchKernel(act_p_gate_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemFreeAsync(theta_ptr, 0));
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

    void act_cx_gate(UINT control, UINT target)
    {
        static double matrix_re[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        static double matrix_im[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        act_two_qubit_gate(control, target, matrix_re, matrix_im);
    }

    void act_cz_gate(UINT target, UINT control)
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

    void act_cz_gate_opt(UINT control, UINT target)
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
        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, dice_ptr_));
        VEDA(vedaArgsSetVPtr(args, 3, x_samples_ptr_));
        VEDA(vedaArgsSetVPtr(args, 4, y_samples_ptr_));
        VEDA(vedaArgsSetVPtr(args, 5, z_samples_ptr_));
        VEDA(vedaArgsSetF64(args, 6, prob));
        VEDA(vedaArgsSetU64(args, 7, target));
        VEDA(vedaArgsSetU64(args, 8, batch_size_));
        VEDA(vedaArgsSetU64(args, 9, n_));
        VEDA(vedaLaunchKernel(act_depolarizing_gate_1q_, 0, args));
        VEDA(vedaArgsDestroy(args));
    }

    void act_depolarizing_gate_2q(UINT control, UINT target, double prob)
    {
        act_depolarizing_gate_1q(control, 1.0 - std::sqrt(1.0 - prob));
        act_depolarizing_gate_1q(target, 1.0 - std::sqrt(1.0 - prob));
    }

    std::vector<std::complex<double>> observe(const Observable &obs) const
    {
        VEDAdeviceptr state_re_ptr, state_im_ptr;
        VEDAdeviceptr bit_flip_mask_ptr, phase_flip_mask_ptr, coef_ptr, expectation_ptr;

        std::vector<UINT> bit_flip_masks(batch_size_);
        std::vector<UINT> phase_flip_masks(batch_size_);
        std::vector<std::complex<double>> coefs(batch_size_);
        std::vector<std::complex<double>> expectations(batch_size_);

        for (int i = 0; i < obs.terms.size(); i++) {
            bit_flip_masks[i] = obs.terms[i].bit_flip_mask;
            phase_flip_masks[i] = obs.terms[i].phase_flip_mask;
            coefs[i] = obs.terms[i].coef;
        }

        VEDA(vedaMemAllocAsync(&bit_flip_mask_ptr, batch_size_ * sizeof(UINT), 0));
        VEDA(vedaMemAllocAsync(&phase_flip_mask_ptr, batch_size_ * sizeof(UINT), 0));
        VEDA(vedaMemAllocAsync(&coef_ptr, batch_size_ * sizeof(std::complex<double>), 0));
        VEDA(vedaMemAllocAsync(&expectation_ptr, batch_size_ * sizeof(std::complex<double>), 0));
        VEDA(vedaMemcpyHtoDAsync(bit_flip_mask_ptr, bit_flip_masks.data(),
                                batch_size_ * sizeof(UINT), 0));
        VEDA(vedaMemcpyHtoDAsync(phase_flip_mask_ptr, phase_flip_masks.data(),
                                batch_size_ * sizeof(UINT), 0));
        VEDA(vedaMemcpyHtoDAsync(coef_ptr, coefs.data(), batch_size_ * sizeof(std::complex<double>),
                                0));

        VEDAargs args;
        VEDA(vedaArgsCreate(&args));
        VEDA(vedaArgsSetVPtr(args, 0, state_re_ptr_));
        VEDA(vedaArgsSetVPtr(args, 1, state_im_ptr_));
        VEDA(vedaArgsSetVPtr(args, 2, bit_flip_mask_ptr));
        VEDA(vedaArgsSetVPtr(args, 3, phase_flip_mask_ptr));
        VEDA(vedaArgsSetVPtr(args, 4, coef_ptr));
        VEDA(vedaArgsSetVPtr(args, 5, expectation_ptr));
        VEDA(vedaArgsSetU64(args, 6, obs.terms.size()));
        VEDA(vedaArgsSetU64(args, 7, batch_size_));
        VEDA(vedaArgsSetU64(args, 8, n_));
        VEDA(vedaLaunchKernel(observe_, 0, args));
        VEDA(vedaArgsDestroy(args));

        VEDA(vedaMemcpyDtoHAsync(expectations.data(), expectation_ptr,
                                 batch_size_ * sizeof(std::complex<double>), 0));

        VEDA(vedaMemFreeAsync(bit_flip_mask_ptr, 0));
        VEDA(vedaMemFreeAsync(phase_flip_mask_ptr, 0));
        VEDA(vedaMemFreeAsync(coef_ptr, 0));
        VEDA(vedaMemFreeAsync(expectation_ptr, 0));
        VEDA(vedaStreamSynchronize(0));

        return expectations;
    }

    void synchronize() { VEDA(vedaCtxSynchronize()); }

private:
    UINT batch_size_;
    UINT n_;

    std::random_device seed_gen_;
    std::mt19937 mt_engine_;
    std::uniform_real_distribution<double> dist_;

    VEDAdeviceptr state_re_ptr_, state_im_ptr_;
    VEDAdeviceptr dice_ptr_, x_samples_ptr_, y_samples_ptr_, z_samples_ptr_;

    static VEDAcontext context_;
    static VEDAmodule mod_;

    static VEDAfunction get_vector_;
    static VEDAfunction get_probability_average_;
    static VEDAfunction get_probability_batched_;
    static VEDAfunction set_zero_state_;
    static VEDAfunction act_single_qubit_gate_;
    static VEDAfunction act_two_qubit_gate_;
    static VEDAfunction act_rx_gate_;
    static VEDAfunction act_ry_gate_;
    static VEDAfunction act_rz_gate_;
    static VEDAfunction act_p_gate_;
    static VEDAfunction act_x_gate_opt_;
    static VEDAfunction act_y_gate_opt_;
    static VEDAfunction act_z_gate_opt_;
    static VEDAfunction act_cx_gate_opt_;
    static VEDAfunction act_cz_gate_opt_;
    static VEDAfunction act_depolarizing_gate_1q_;
    static VEDAfunction observe_;
};

VEDAcontext State::Impl::context_;
VEDAmodule State::Impl::mod_;

VEDAfunction State::Impl::get_vector_;
VEDAfunction State::Impl::get_probability_average_;
VEDAfunction State::Impl::get_probability_batched_;
VEDAfunction State::Impl::set_zero_state_;
VEDAfunction State::Impl::act_single_qubit_gate_;
VEDAfunction State::Impl::act_two_qubit_gate_;
VEDAfunction State::Impl::act_rx_gate_;
VEDAfunction State::Impl::act_ry_gate_;
VEDAfunction State::Impl::act_rz_gate_;
VEDAfunction State::Impl::act_p_gate_;
VEDAfunction State::Impl::act_x_gate_opt_;
VEDAfunction State::Impl::act_y_gate_opt_;
VEDAfunction State::Impl::act_z_gate_opt_;
VEDAfunction State::Impl::act_cx_gate_opt_;
VEDAfunction State::Impl::act_cz_gate_opt_;
VEDAfunction State::Impl::act_depolarizing_gate_1q_;
VEDAfunction State::Impl::observe_;

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

void State::synchronize() { impl_->synchronize(); }

void initialize() { State::initialize(); }

void finalize() { State::finalize(); }

}
