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

TEST_CASE("H gate")
{
    const int N = 4;
    const int BATCH_SIZE = 10;

    std::vector<double> state_re((1ULL << N) * BATCH_SIZE), state_im((1ULL << N) * BATCH_SIZE);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        state_re[sample + 0 * BATCH_SIZE] = 1.0;
        state_im[sample + 0 * BATCH_SIZE] = 0.0;
    }

    for (int i = 0; i < N; i++) {
        apply_h_gate(state_re, state_im, BATCH_SIZE, N, i);
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        for (int i = 0; i < N; i++) {
            REQUIRE(state_re[sample + i * BATCH_SIZE]== doctest::Approx(0.25));
            REQUIRE(state_im[sample + i * BATCH_SIZE]== doctest::Approx(0.0));
        }
    }
}


TEST_CASE("CZ gate")
{
    const int N = 3;
    const int BATCH_SIZE = 10;

    std::vector<double> state_re((1ULL << N) * BATCH_SIZE), state_im((1ULL << N) * BATCH_SIZE);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        state_re[sample + 0 * BATCH_SIZE] = 1.0;
        state_im[sample + 0 * BATCH_SIZE] = 0.0;
    }

    for (int i = 0; i < N; i++) {
        apply_h_gate(state_re, state_im, BATCH_SIZE, N, i);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            apply_cz_gate(state_re, state_im, BATCH_SIZE, N, j, i);
        }
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(state_re[sample + 0 * BATCH_SIZE] == doctest::Approx(0.3535533905932737));
        REQUIRE(state_im[sample + 0 * BATCH_SIZE] == doctest::Approx(0.0));
        REQUIRE(state_re[sample + 1 * BATCH_SIZE] == doctest::Approx(0.3535533905932737));
        REQUIRE(state_im[sample + 1 * BATCH_SIZE] == doctest::Approx(0.0));
        REQUIRE(state_re[sample + 2 * BATCH_SIZE] == doctest::Approx(0.3535533905932737));
        REQUIRE(state_im[sample + 2 * BATCH_SIZE] == doctest::Approx(0.0));
        REQUIRE(state_re[sample + 3 * BATCH_SIZE] == doctest::Approx(-0.3535533905932737));
        REQUIRE(state_im[sample + 3 * BATCH_SIZE] == doctest::Approx(-0.0));
        REQUIRE(state_re[sample + 4 * BATCH_SIZE] == doctest::Approx(0.3535533905932737));
        REQUIRE(state_im[sample + 4 * BATCH_SIZE] == doctest::Approx(0.0));
        REQUIRE(state_re[sample + 5 * BATCH_SIZE] == doctest::Approx(-0.3535533905932737));
        REQUIRE(state_im[sample + 5 * BATCH_SIZE] == doctest::Approx(-0.0));
        REQUIRE(state_re[sample + 6 * BATCH_SIZE] == doctest::Approx(-0.3535533905932737));
        REQUIRE(state_im[sample + 6 * BATCH_SIZE] == doctest::Approx(-0.0));
        REQUIRE(state_re[sample + 7 * BATCH_SIZE] == doctest::Approx(-0.3535533905932737));
        REQUIRE(state_im[sample + 7 * BATCH_SIZE] == doctest::Approx(-0.0));
    }
}
