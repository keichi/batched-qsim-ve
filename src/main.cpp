#include <chrono>
#include <complex>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "gate.hpp"

#define LAYOUT_SOA2

void apply_rx_all_soa2(std::vector<double> &state_re, std::vector<double> &state_im,
                       UINT BATCH_SIZE, UINT N_QUBITS, UINT DEPTH, std::mt19937 &engine,
                       std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            apply_rx_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, dist(engine), i);
        }
    }
}

void apply_rx_all_soa1(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT N_QUBITS,
                       UINT DEPTH, std::mt19937 &engine,
                       std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            apply_rx_gate_soa1(state, BATCH_SIZE, N_QUBITS, dist(engine), i);
        }
    }
}

void apply_rx_all_aos(std::vector<std::complex<double>> &state, UINT BATCH_SIZE, UINT N_QUBITS,
                      UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            apply_rx_gate_aos(state, BATCH_SIZE, N_QUBITS, dist(engine), i);
        }
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("batched-qsim-ve", "Batched quantum circuit simulator for VE");
    // clang-format off
    options.add_options()
        ("samples", "# of samples", cxxopts::value<int>()->default_value("1000"))
        ("target", "Target qubit (round-robin if -1)", cxxopts::value<int>()->default_value("-1"))
        ("depth", "Depth of the circuit", cxxopts::value<int>()->default_value("10"))
        ("qubits", "# of qubits", cxxopts::value<int>()->default_value("10"))
        ("batch-size", "Batch size", cxxopts::value<int>()->default_value("1000"))
        ("trials", "# of trials", cxxopts::value<int>()->default_value("10"));
    // clang-format on

    auto result = options.parse(argc, argv);

    const int N_SAMPLES = result["samples"].as<int>();
    const int TARGET = result["target"].as<int>();
    const int DEPTH = result["depth"].as<int>();
    const int N_QUBITS = result["qubits"].as<int>();
    const int BATCH_SIZE = result["batch-size"].as<int>();
    const int N_TRIALS = result["trials"].as<int>();

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> durations;
#if  defined(LAYOUT_SOA2)
    std::vector<double> state_re((1ULL << N_QUBITS) * BATCH_SIZE);
    std::vector<double> state_im((1ULL << N_QUBITS) * BATCH_SIZE);
#elif defined(LAYOUT_SOA1) || defined(LAYOUT_AOS)
    std::vector<std::complex<double>> state((1ULL << N_QUBITS) * BATCH_SIZE);
#endif

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
#if defined(LAYOUT_SOA1)
        apply_rx_all_soa1(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#elif defined(LAYOUT_SOA2)
        apply_rx_all_soa2(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#elif defined(LAYOUT_AOS)
        apply_rx_all_aos(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#endif
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
#if defined(LAYOUT_SOA1)
            apply_rx_all_soa1(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#elif defined(LAYOUT_SOA2)
            apply_rx_all_soa2(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#elif defined(LAYOUT_AOS)
            apply_rx_all_aos(state, BATCH_SIZE, N_QUBITS, DEPTH, engine, dist);
#endif
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
