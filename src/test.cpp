#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "gate.hpp"

TEST_CASE("RX gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;

    std::vector<double> state_re((1ULL << N) * BATCH_SIZE), state_im((1ULL << N) * BATCH_SIZE);

    for (int i = 0; i < N; i++) {
        apply_rx_gate(state_re, state_im, BATCH_SIZE, N, 1.0, i);
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(state_re[sample + 0 * BATCH_SIZE]== doctest::Approx(0.7701511529340699));
        REQUIRE(state_im[sample + 0 * BATCH_SIZE]== doctest::Approx(0.0));
    }
}
