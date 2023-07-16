#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "gate.hpp"

TEST_CASE("RX gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;

    std::vector<double> state_re((1ULL << N) * BATCH_SIZE), state_im((1ULL << N) * BATCH_SIZE);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        state_re[sample + 0 * BATCH_SIZE] = 1.0;
        state_im[sample + 0 * BATCH_SIZE] = 0.0;
    }

    for (int i = 0; i < N; i++) {
        apply_rx_gate(state_re, state_im, BATCH_SIZE, N, 1.0, i);
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(state_re[sample + 0 * BATCH_SIZE]== doctest::Approx(0.7701511529340699));
        REQUIRE(state_im[sample + 0 * BATCH_SIZE]== doctest::Approx(0.0));
        REQUIRE(state_re[sample + 1 * BATCH_SIZE]== doctest::Approx(0.0));
        REQUIRE(state_im[sample + 1 * BATCH_SIZE]== doctest::Approx(-0.42073549240394825));
        REQUIRE(state_re[sample + 2 * BATCH_SIZE]== doctest::Approx(0.0));
        REQUIRE(state_im[sample + 2 * BATCH_SIZE]== doctest::Approx(-0.42073549240394825));
        REQUIRE(state_re[sample + 3 * BATCH_SIZE]== doctest::Approx(-0.22984884706593015));
        REQUIRE(state_im[sample + 3 * BATCH_SIZE]== doctest::Approx(0.0));
    }
}
