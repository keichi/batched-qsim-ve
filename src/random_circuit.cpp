#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <cxxopts.hpp>

#include "gate.hpp"


UINT grid_to_id(UINT x, UINT y, UINT length)
{
    return x + length * y;
}

bool in_grid(UINT x, UINT y, UINT length)
{
    return x < length && y < length && x >= 0 && y >= 0;
}

void apply_2q_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                   UINT BATCH_SIZE, UINT LENGTH, UINT x1, UINT y1, UINT x2, UINT y2)
{
    if (!in_grid(x1, y1, LENGTH) || !in_grid(x2, y2, LENGTH)) {
        return;
    }

    UINT N_QUBITS  = LENGTH * LENGTH;
    static double theta = M_PI / 2;
    UINT target = grid_to_id(x1, y1, LENGTH);
    UINT control = grid_to_id(x2, y2, LENGTH);

    apply_iswaplike_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, theta, target, control);
}

void apply_random_1q_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                          UINT BATCH_SIZE, UINT LENGTH, double dice, UINT target)
{
    UINT N_QUBITS  = LENGTH * LENGTH;

    if (dice < 1.0 / 3.0) {
        apply_sx_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, target);
    } else if (dice < 2.0 / 3.0) {
        apply_sy_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, target);
    } else {
        apply_sw_gate(state_re, state_im, BATCH_SIZE, N_QUBITS, target);
    }
}

void run_single_batch(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                      UINT LENGTH, UINT DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist)
{
    UINT N_QUBITS  = LENGTH * LENGTH;

    for (int d = 0; d < DEPTH; d++) {
        if (d % 4 == 0) {
            for (int i = 0; i < N_QUBITS; i++) {
                apply_random_1q_gate(state_re, state_im, BATCH_SIZE, LENGTH, dist(engine), i);
            }
        } else if (d % 4 == 1) {
            for (int i = 0; i < LENGTH; i++) {
                apply_random_1q_gate(state_re, state_im, BATCH_SIZE, LENGTH, dist(engine), i);
            }
        } else if (d % 4 == 2) {
            for (int i = 0; i < LENGTH; i++) {
                apply_random_1q_gate(state_re, state_im, BATCH_SIZE, LENGTH, dist(engine), i);
            }
        } else if (d % 4 == 3) {
            for (int i = 0; i < LENGTH; i++) {
                apply_random_1q_gate(state_re, state_im, BATCH_SIZE, LENGTH, dist(engine), i);
            }
        }

        for (int i = 0; i < LENGTH; i++) {
            for (int j = 0; j < LENGTH; j++) {
                if ((i + j) % 2 != 0) {
                    continue;
                }

                if (d % 4 == 0) {
                    apply_2q_gate(state_re, state_im, BATCH_SIZE, LENGTH, i, j, i + 1, j);
                } else if (d % 4 == 1) {
                    apply_2q_gate(state_re, state_im, BATCH_SIZE, LENGTH, i, j, i - 1, j);
                } else if (d % 4 == 2) {
                    apply_2q_gate(state_re, state_im, BATCH_SIZE, LENGTH, i, j, i, j + 1);
                } else if (d % 4 == 3) {
                    apply_2q_gate(state_re, state_im, BATCH_SIZE, LENGTH, i, j, i, j - 1);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("batched-qsim-ve", "Batched quantum circuit simulator for VE");
    // clang-format off
    options.add_options()
        ("samples", "# of samples", cxxopts::value<int>()->default_value("1000"))
        ("depth", "Depth of the circuit", cxxopts::value<int>()->default_value("10"))
        ("length", "Side length of grid", cxxopts::value<int>()->default_value("4"))
        ("batch-size", "Batch size", cxxopts::value<int>()->default_value("1000"))
        ("trials", "# of trials", cxxopts::value<int>()->default_value("10"));
    // clang-format on

    auto result = options.parse(argc, argv);

    const int N_SAMPLES = result["samples"].as<int>();
    const int DEPTH = result["depth"].as<int>();
    const int LENGTH = result["length"].as<int>();
    const int BATCH_SIZE = result["batch-size"].as<int>();
    const int N_TRIALS = result["trials"].as<int>();

    const int N_QUBITS = LENGTH * LENGTH;

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0, 1);
    std::vector<double> durations;
    std::vector<double> state_re((1ULL << N_QUBITS) * BATCH_SIZE);
    std::vector<double> state_im((1ULL << N_QUBITS) * BATCH_SIZE);

    set_zero_state(state_re, state_im, BATCH_SIZE, N_QUBITS);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        run_single_batch(state_re, state_im, BATCH_SIZE, LENGTH, DEPTH, engine, dist);
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        set_zero_state(state_re, state_im, BATCH_SIZE, N_QUBITS);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            run_single_batch(state_re, state_im, BATCH_SIZE, LENGTH, DEPTH, engine, dist);
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
