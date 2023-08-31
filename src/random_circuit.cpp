#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "gate.hpp"

UINT grid_to_id(UINT x, UINT y, UINT length) { return x + length * y; }

bool in_grid(UINT x, UINT y, UINT length) { return x < length && y < length; }

void act_2q_gate(State &state, UINT LENGTH, UINT x1, UINT y1, UINT x2, UINT y2, double noise_rate)
{
    if (!in_grid(x1, y1, LENGTH) || !in_grid(x2, y2, LENGTH)) {
        return;
    }

    static double theta = M_PI / 2;
    UINT target = grid_to_id(x1, y1, LENGTH);
    UINT control = grid_to_id(x2, y2, LENGTH);

    state.act_iswaplike_gate(theta, target, control);

    if (noise_rate) {
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

    if (noise_rate > 0.0) {
        state.act_depolarizing_gate_1q(target, noise_rate);
    }
}

void run_single_batch(State &state, UINT LENGTH, UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist, double noise_rate)
{
    UINT N_QUBITS = LENGTH * LENGTH;

    for (int d = 0; d < DEPTH; d++) {
        double dice = dist(engine);

        for (int i = 0; i < N_QUBITS; i++) {
            act_random_1q_gate(state, dice, i, noise_rate);
        }

        for (int i = 0; i < LENGTH; i++) {
            for (int j = 0; j < LENGTH; j++) {
                if ((i + j) % 2 != 0) {
                    continue;
                }

                if (d % 4 == 0) {
                    act_2q_gate(state, LENGTH, i, j, i + 1, j, noise_rate);
                } else if (d % 4 == 1) {
                    act_2q_gate(state, LENGTH, i, j, i - 1, j, noise_rate);
                } else if (d % 4 == 2) {
                    act_2q_gate(state, LENGTH, i, j, i, j + 1, noise_rate);
                } else if (d % 4 == 3) {
                    act_2q_gate(state, LENGTH, i, j, i, j - 1, noise_rate);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("qsim-random-circuit", "Noisy random circuit simulator for VE");
    // clang-format off
    options.add_options()
        ("samples", "# of samples", cxxopts::value<int>()->default_value("1000"))
        ("depth", "Depth of the circuit", cxxopts::value<int>()->default_value("10"))
        ("length", "Side length of grid", cxxopts::value<int>()->default_value("4"))
        ("batch-size", "Batch size", cxxopts::value<int>()->default_value("1000"))
        ("trials", "# of trials", cxxopts::value<int>()->default_value("10"))
        ("noise-rate", "Noise rate", cxxopts::value<double>()->default_value("0.1"));
    // clang-format on

    auto result = options.parse(argc, argv);

    const int N_SAMPLES = result["samples"].as<int>();
    const int DEPTH = result["depth"].as<int>();
    const int LENGTH = result["length"].as<int>();
    const int BATCH_SIZE = result["batch-size"].as<int>();
    const int N_TRIALS = result["trials"].as<int>();
    const double NOISE_RATE = result["noise-rate"].as<double>();
    const int N_QUBITS = LENGTH * LENGTH;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<double> durations;
    State state(N_QUBITS, BATCH_SIZE);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        state.set_zero_state();
        run_single_batch(state, LENGTH, DEPTH, engine, dist, NOISE_RATE);
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        state.set_zero_state();

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            run_single_batch(state, LENGTH, DEPTH, engine, dist, NOISE_RATE);
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

    std::cout << average << std::endl;

    return 0;
}
