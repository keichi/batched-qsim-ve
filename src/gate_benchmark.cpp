#include <chrono>
#include <complex>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "state.hpp"

int main(int argc, char *argv[])
{
    cxxopts::Options options("qsim-gate-bench", "Basic gate benchmark");
    // clang-format off
    options.add_options()
        ("samples", "# of samples", cxxopts::value<int>()->default_value("1000"))
        ("target", "Target qubit", cxxopts::value<int>()->default_value("0"))
        ("control", "Control qubit", cxxopts::value<int>()->default_value("1"))
        ("depth", "Depth of the circuit", cxxopts::value<int>()->default_value("10"))
        ("qubits", "# of qubits", cxxopts::value<int>()->default_value("10"))
        ("batch-size", "Batch size", cxxopts::value<int>()->default_value("1000"))
        ("trials", "# of trials", cxxopts::value<int>()->default_value("10"))
        ("gate", "Gate to act", cxxopts::value<std::string>()->default_value("RX"))
        ("help", "Print usage");
    // clang-format on

    auto result = options.parse(argc, argv);

    int n_samples = result["samples"].as<int>();
    int target = result["target"].as<int>();
    int control = result["control"].as<int>();
    int depth = result["depth"].as<int>();
    int n_qubits = result["qubits"].as<int>();
    int batch_size = result["batch-size"].as<int>();
    int n_trials = result["trials"].as<int>();
    std::string gate_name = result["gate"].as<std::string>();

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0.0, M_PI * 2);
    std::vector<double> durations;
    State state(n_qubits, batch_size);

    // Warmup run
    for (int batch = 0; batch < n_samples; batch += batch_size) {
        for (int d = 0; d < depth; d++) {
            if (gate_name == "RX") {
                state.act_rx_gate(dist(engine), target);
            } else if (gate_name == "H") {
                state.act_h_gate(target);
            } else if (gate_name == "T") {
                state.act_t_gate(target);
            } else if (gate_name == "CNOT") {
                state.act_cnot_gate(target, control);
            }
        }
    }

    for (int trial = 0; trial < n_trials; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < n_samples; batch += batch_size) {
            for (int d = 0; d < depth; d++) {
                if (gate_name == "RX") {
                    state.act_rx_gate(dist(engine), target);
                } else if (gate_name == "H") {
                    state.act_h_gate(target);
                } else if (gate_name == "T") {
                    state.act_t_gate(target);
                } else if (gate_name == "CNOT") {
                    state.act_cnot_gate(target, control);
                }
            }
        }

        state.synchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        durations.push_back(duration.count() / 1e6);
    }

    double average = 0.0;
    for (int trial = 0; trial < n_trials; trial++) {
        average += durations[trial];
    }
    average /= n_trials;

    double variance = 0.0;
    for (int trial = 0; trial < n_trials; trial++) {
        variance += (durations[trial] - average) * (durations[trial] - average);
    }
    variance = variance / n_trials;

    std::cout << average << std::endl;

    return 0;
}
