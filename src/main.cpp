#include <chrono>
#include <complex>
#include <iostream>
#include <random>

using UINT = unsigned int;
using ITYPE = unsigned long long;

void update_with_RX_batched(std::vector<double> &state_r,
                            std::vector<double> &state_i, UINT BATCH_SIZE,
                            UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half),
           cos_half = std::cos(angle_half);

#pragma omp parallel for
    for (ITYPE i = 0; i < 1ULL << (n - 1); i++) {
#pragma _NEC ivdep
        for (int sample = 0; sample < BATCH_SIZE; sample++) {
            ITYPE j = i | (1ULL << target);

            double tmp_i_r = state_r[sample + i * BATCH_SIZE];
            double tmp_i_i = state_i[sample + i * BATCH_SIZE];
            double tmp_j_r = state_r[sample + j * BATCH_SIZE];
            double tmp_j_i = state_i[sample + j * BATCH_SIZE];

            // state[i] = cos(t/2) * state[i] + i * sin(t/2) * state[j]
            state_r[sample + i * BATCH_SIZE] =
                cos_half * tmp_i_r - sin_half * tmp_j_i;
            state_i[sample + i * BATCH_SIZE] =
                cos_half * tmp_i_i + sin_half * tmp_j_r;

            // state[j] = -i * sin(t/2) * state[i] + i * cos(t/2) * state[j]
            state_r[sample + j * BATCH_SIZE] =
                -sin_half * tmp_i_i + cos_half * tmp_j_r;
            state_i[sample + j * BATCH_SIZE] =
                sin_half * tmp_i_r + cos_half * tmp_j_i;
        }
    }
}

void run_single_batch(std::vector<double> &state_r,
                      std::vector<double> &state_i, UINT BATCH_SIZE,
                      UINT N_QUBITS, UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            update_with_RX_batched(state_r, state_i, BATCH_SIZE, N_QUBITS,
                                   dist(engine), i);
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
    std::vector<double> state_r((1ULL << N_QUBITS) * BATCH_SIZE);
    std::vector<double> state_i((1ULL << N_QUBITS) * BATCH_SIZE);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        run_single_batch(state_r, state_i, BATCH_SIZE, N_QUBITS, DEPTH, engine,
                         dist);
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            run_single_batch(state_r, state_i, BATCH_SIZE, N_QUBITS, DEPTH,
                             engine, dist);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
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
