#include <chrono>
#include <complex>
#include <iostream>
#include <random>

using UINT = unsigned int;
using ITYPE = unsigned long long;

void apply_single_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                             UINT BATCH_SIZE, UINT n, const double matrix_re[2][2],
                             const double matrix_im[2][2], UINT target)
{
#pragma omp parallel for
    for (ITYPE i0 = 0; i0 < 1ULL << (n - 1); i0++) {
            ITYPE i1 = i0 | (1ULL << target);

#pragma _NEC ivdep
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

void apply_rx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, double angle, UINT target)
{
    double matrix_re[2][2] = {{std::cos(angle / 2), 0}, {0, std::cos(angle / 2)}};
    double matrix_im[2][2] = {{0, -std::sin(angle / 2)}, {-std::sin(angle / 2), 0}};

    apply_single_qubit_gate(state_re, state_im, BATCH_SIZE, n, matrix_re, matrix_im, target);
}

void run_single_batch(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                      UINT N_QUBITS, UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            apply_rx_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, dist(engine), i);
        }
    }
}

int main(int argc, char *argv[])
{
    const int N_SAMPLES = std::atoi(argv[2]);
    const int DEPTH = 10;
    const int N_QUBITS = std::atoi(argv[1]);
    const int BATCH_SIZE = std::atoi(argv[2]);
    const int N_TRIALS = 10;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> durations;
    std::vector<double> state_re((1ULL << N_QUBITS) * BATCH_SIZE);
    std::vector<double> state_im((1ULL << N_QUBITS) * BATCH_SIZE);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        run_single_batch(state_re, state_im, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            run_single_batch(state_re, state_im, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        durations.push_back(duration.count() / 1e6);
    }

    double average = 0.0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        average += durations[trial];
    }
    average /= N_TRIALS;

    double variance = 0.0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        variance += (durations[trial] - average) * (durations[trial] - average);
    }
    variance = variance / N_TRIALS;

    std::cout << average << std::endl;

    return 0;
}
