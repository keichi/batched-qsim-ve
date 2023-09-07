#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "state.hpp"

UINT grid_to_id(UINT x, UINT y, UINT width) { return x + width * y; }

bool in_grid(UINT x, UINT y, UINT width, UINT height) { return x < width && y < height; }

void act_2q_gate(State &state, UINT width, UINT height, UINT x1, UINT y1, UINT x2, UINT y2,
                 double noise_rate)
{
    if (!in_grid(x1, y1, width, height) || !in_grid(x2, y2, width, height)) {
        return;
    }

    static double theta = M_PI / 2;
    UINT target = grid_to_id(x1, y1, width);
    UINT control = grid_to_id(x2, y2, width);

    state.act_iswaplike_gate(theta, target, control);

    if (noise_rate > 0) {
        state.act_depolarizing_gate_2q(target, control, noise_rate);
    }
}

void act_random_1q_gate(State &state, double dice, UINT target, double noise_rate)
{
    if (dice < 1.0 / 3.0) {
        state.act_sx_gate(target);
    } else if (dice < 2.0 / 3.0) {
        state.act_sy_gate(target);
    } else {
        state.act_sw_gate(target);
    }

    if (noise_rate > 0) {
        state.act_depolarizing_gate_1q(target, noise_rate);
    }
}

void run_single_batch(State &state, UINT width, UINT height, UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist, double noise_rate)
{
    UINT N_QUBITS = width * height;

    for (int d = 0; d < DEPTH; d++) {
        double dice = dist(engine);

        for (int i = 0; i < N_QUBITS; i++) {
            act_random_1q_gate(state, dice, i, noise_rate);
        }

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if ((i + j) % 2 != 0) {
                    continue;
                }

                if (d % 4 == 0) {
                    act_2q_gate(state, width, height, i, j, i + 1, j, noise_rate);
                } else if (d % 4 == 1) {
                    act_2q_gate(state, width, height, i, j, i - 1, j, noise_rate);
                } else if (d % 4 == 2) {
                    act_2q_gate(state, width, height, i, j, i, j + 1, noise_rate);
                } else if (d % 4 == 3) {
                    act_2q_gate(state, width, height, i, j, i, j - 1, noise_rate);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("qsim-random-circuit", "Noisy random circuit benchmark");
    // clang-format off
    options.add_options()
        ("samples", "# of samples", cxxopts::value<int>()->default_value("1000"))
        ("depth", "Depth of the circuit", cxxopts::value<int>()->default_value("10"))
        ("width", "Width of grid", cxxopts::value<int>()->default_value("4"))
        ("height", "Height of grid", cxxopts::value<int>()->default_value("4"))
        ("batch-size", "Batch size", cxxopts::value<int>()->default_value("1000"))
        ("trials", "# of trials", cxxopts::value<int>()->default_value("10"))
        ("noise-rate", "Noise rate", cxxopts::value<double>()->default_value("0.1"))
        ("help", "Print usage");
    // clang-format on

    auto result = options.parse(argc, argv);

    int n_samples = result["samples"].as<int>();
    int depth = result["depth"].as<int>();
    int width = result["width"].as<int>();
    int height = result["height"].as<int>();
    int batch_size = result["batch-size"].as<int>();
    int n_trials = result["trials"].as<int>();
    double noise_rate = result["noise-rate"].as<double>();
    int n_qubits = width * height;

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<double> durations;
    State state(n_qubits, batch_size);

    // Warmup run
    for (int batch = 0; batch < n_samples; batch += batch_size) {
        state.set_zero_state();
        run_single_batch(state, width, height, depth, engine, dist, noise_rate);
    }

    for (int trial = 0; trial < n_trials; trial++) {
        state.set_zero_state();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < n_samples; batch += batch_size) {
            run_single_batch(state, width, height, depth, engine, dist, noise_rate);
        }

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

    std::cout << average << std::endl;

    return 0;
}
