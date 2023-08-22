#include <chrono>
#include <complex>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "gate.hpp"

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
    State state(N_QUBITS, BATCH_SIZE);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        for (int d = 0; d < DEPTH; d++) {
            for (int i = 0; i < N_QUBITS; i++) {
                state.apply_rx_gate(dist(engine), TARGET);
            }
            for (int i = 0; i < N_QUBITS; i++) {
                state.apply_cnot_gate(TARGET, (TARGET + 1) % N_QUBITS);
            }
        }
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            for (int d = 0; d < DEPTH; d++) {
                for (int i = 0; i < N_QUBITS; i++) {
                    state.apply_rx_gate(dist(engine), TARGET);
                }
                for (int i = 0; i < N_QUBITS; i++) {
                    state.apply_cnot_gate(TARGET, (TARGET + 1) % N_QUBITS);
                }
            }
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
