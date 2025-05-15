#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../src/state.hpp"

namespace nb = nanobind;

NB_MODULE(_veqsim, m)
{
    nb::enum_<PauliID>(m, "PauliID")
        .value("I", PauliID::I)
        .value("X", PauliID::X)
        .value("Y", PauliID::Y)
        .value("Z", PauliID::Z);

    nb::class_<PauliOperator>(m, "PauliOperator")
        .def(nb::init<std::complex<double>, std::vector<UINT>, std::vector<UINT>>());

    nb::class_<Observable>(m, "Observable")
        .def(nb::init<>())
        .def("add_operator", &Observable::add_operator);

    nb::class_<State>(m, "State", "A batched state vector")
        .def(nb::init<UINT, UINT>(), "Create state vector")
        .def_prop_ro("dim", &State::dim, "Dimension")
        .def_prop_ro("batch_size", &State::batch_size, "Batch size")
        .def("get_vector", &State::get_vector, "Get the state vector of a sample")
        .def("amplitude", &State::amplitude, "Get the amplitude of a basis")
        .def("re", &State::re, "Get the real component of the amplitude of a basis")
        .def("im", &State::im, "Get the imaginary component of the amplitude of a basis")
        .def("get_probability", &State::get_probability, "Get ")
        .def("set_zero_state", &State::set_zero_state, "Initialize to |0..0>")
        .def("act_x_gate", &State::act_x_gate, "Apply an X gate")
        .def("act_y_gate", &State::act_y_gate, "Apply a Y gate")
        .def("act_z_gate", &State::act_z_gate, "Apply a Z gate")
        .def("act_h_gate", &State::act_h_gate, "Apply an H gate")
        .def("act_t_gate", &State::act_t_gate, "Apply a T gate")
        .def("act_rx_gate", nb::overload_cast<double, UINT>(&State::act_rx_gate),
             "Act an RX gate with the same angle")
        .def("act_rx_gate",
             nb::overload_cast<const std::vector<double> &, UINT>(&State::act_rx_gate),
             "Act an RX gate with different angles")
        .def("act_ry_gate", &State::act_ry_gate, "Apply an RY gate")
        .def("act_rz_gate", &State::act_rz_gate, "Apply an RZ gate")
        .def("act_sx_gate", &State::act_sx_gate, "Apply an SX gate")
        .def("act_sy_gate", &State::act_sy_gate, "Apply an SY gate")
        .def("act_sw_gate", &State::act_sw_gate, "Apply an SW gate")
        .def("act_cnot_gate", &State::act_cnot_gate, "Apply a CNOT gate")
        .def("act_cx_gate", &State::act_cx_gate, "Apply a CX gate")
        .def("act_cz_gate", &State::act_cz_gate, "Apply a CZ gate")
        .def("act_iswaplike_gate", &State::act_iswaplike_gate, "Apply an iSWAP-like gate")
        .def("act_depolarizing_gate_1q", &State::act_depolarizing_gate_1q,
             "Apply an one-qubit depolarizing noise gate")
        .def("act_depolarizing_gate_2q", &State::act_depolarizing_gate_2q,
             "Apply a two-qubit depolarizing noise gate")
        .def("observe", &State::observe, "Get expectation of a given observable")
        .def("synchronize", &State::synchronize, "Wiat for completion of computation");
}
