#include <chrono>
#include <iostream>
#include <random>

#include "gate.hpp"

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
