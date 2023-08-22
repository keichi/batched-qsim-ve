#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "gate.hpp"

TEST_CASE("RX gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;

    State state(N, BATCH_SIZE);

    state.set_zero_state();

    for (int i = 0; i < N; i++) {
        state.act_rx_gate(1.0, i);
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(state.re(sample, 0) == doctest::Approx(0.7701511529340699));
        REQUIRE(state.im(sample, 0) == doctest::Approx(0.0));
        REQUIRE(state.re(sample, 1) == doctest::Approx(0.0));
        REQUIRE(state.im(sample, 1) == doctest::Approx(-0.42073549240394825));
        REQUIRE(state.re(sample, 2) == doctest::Approx(0.0));
        REQUIRE(state.im(sample, 2) == doctest::Approx(-0.42073549240394825));
        REQUIRE(state.re(sample, 3) == doctest::Approx(-0.22984884706593015));
        REQUIRE(state.im(sample, 3) == doctest::Approx(0.0));
    }
}

TEST_CASE("H gate")
{
    const int N = 4;
    const int BATCH_SIZE = 10;

    State state(N, BATCH_SIZE);

    state.set_zero_state();

    for (int i = 0; i < N; i++) {
        state.act_h_gate(i);
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        for (int i = 0; i < N; i++) {
            REQUIRE(state.re(sample, i)== doctest::Approx(0.25));
            REQUIRE(state.im(sample, i)== doctest::Approx(0.0));
        }
    }
}


TEST_CASE("CZ gate")
{
    const int N = 3;
    const int BATCH_SIZE = 10;

    State state(N, BATCH_SIZE);

    state.set_zero_state();

    for (int i = 0; i < N; i++) {
        state.act_h_gate(i);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            state.act_cz_gate(j, i);
        }
    }

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(state.re(sample, 0) == doctest::Approx(0.3535533905932737));
        REQUIRE(state.im(sample, 0) == doctest::Approx(0.0));
        REQUIRE(state.re(sample, 1) == doctest::Approx(0.3535533905932737));
        REQUIRE(state.im(sample, 1) == doctest::Approx(0.0));
        REQUIRE(state.re(sample, 2) == doctest::Approx(0.3535533905932737));
        REQUIRE(state.im(sample, 2) == doctest::Approx(0.0));
        REQUIRE(state.re(sample, 3) == doctest::Approx(-0.3535533905932737));
        REQUIRE(state.im(sample, 3) == doctest::Approx(-0.0));
        REQUIRE(state.re(sample, 4) == doctest::Approx(0.3535533905932737));
        REQUIRE(state.im(sample, 4) == doctest::Approx(0.0));
        REQUIRE(state.re(sample, 5) == doctest::Approx(-0.3535533905932737));
        REQUIRE(state.im(sample, 5) == doctest::Approx(-0.0));
        REQUIRE(state.re(sample, 6) == doctest::Approx(-0.3535533905932737));
        REQUIRE(state.im(sample, 6) == doctest::Approx(-0.0));
        REQUIRE(state.re(sample, 7) == doctest::Approx(-0.3535533905932737));
        REQUIRE(state.im(sample, 7) == doctest::Approx(-0.0));
    }
}
