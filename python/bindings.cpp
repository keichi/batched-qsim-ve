#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../src/state.hpp"

namespace nb = nanobind;
namespace vq = veqsim;

NB_MODULE(_veqsim, m)
{
    nb::enum_<vq::PauliID>(m, "PauliID")
        .value("I", vq::PauliID::I)
        .value("X", vq::PauliID::X)
        .value("Y", vq::PauliID::Y)
        .value("Z", vq::PauliID::Z);

    nb::class_<vq::PauliOperator>(m, "PauliOperator")
        .def(nb::init<std::complex<double>, std::vector<UINT>, std::vector<UINT>>());

    nb::class_<vq::Observable>(m, "Observable")
        .def(nb::init<>())
        .def("add_operator", &vq::Observable::add_operator);

    nb::class_<vq::State>(m, "State", "A batched state vector")
        .def(nb::init<UINT, UINT>(), "Create state vector")
        .def_prop_ro("dim", &vq::State::dim, "Dimension")
        .def_prop_ro("batch_size", &vq::State::batch_size, "Batch size")
        .def("get_vector", &vq::State::get_vector, "Get a single state vector")
        .def("amplitude", &vq::State::amplitude, "Get the amplitude of a basis")
        .def("re", &vq::State::re, "Get the real component of the amplitude of a basis")
        .def("im", &vq::State::im, "Get the imaginary component of the amplitude of a basis")
        .def("get_probability", &vq::State::get_probability, "Get ")
        .def("set_zero_state", &vq::State::set_zero_state, "Initialize to |0..0>")
        .def("act_x_gate", &vq::State::act_x_gate, "Apply an X gate")
        .def("act_y_gate", &vq::State::act_y_gate, "Apply a Y gate")
        .def("act_z_gate", &vq::State::act_z_gate, "Apply a Z gate")
        .def("act_h_gate", &vq::State::act_h_gate, "Apply an H gate")
        .def("act_t_gate", &vq::State::act_t_gate, "Apply a T gate")
        .def("act_rx_gate", nb::overload_cast<double, UINT>(&vq::State::act_rx_gate),
             "Act an RX gate with the same angle")
        .def("act_rx_gate",
             nb::overload_cast<const std::vector<double> &, UINT>(&vq::State::act_rx_gate),
             "Act an RX gate with different angles")
        .def("act_ry_gate", &vq::State::act_ry_gate, "Apply an RY gate")
        .def("act_rz_gate", &vq::State::act_rz_gate, "Apply an RZ gate")
        .def("act_sx_gate", &vq::State::act_sx_gate, "Apply an SX gate")
        .def("act_sy_gate", &vq::State::act_sy_gate, "Apply an SY gate")
        .def("act_sw_gate", &vq::State::act_sw_gate, "Apply an SW gate")
        .def("act_cnot_gate", &vq::State::act_cnot_gate, "Apply a CNOT gate")
        .def("act_cx_gate", &vq::State::act_cx_gate, "Apply a CX gate")
        .def("act_cz_gate", &vq::State::act_cz_gate, "Apply a CZ gate")
        .def("act_iswaplike_gate", &vq::State::act_iswaplike_gate, "Apply an iSWAP-like gate")
        .def("act_depolarizing_gate_1q", &vq::State::act_depolarizing_gate_1q,
             "Apply an one-qubit depolarizing noise gate")
        .def("act_depolarizing_gate_2q", &vq::State::act_depolarizing_gate_2q,
             "Apply a two-qubit depolarizing noise gate")
        .def("observe", &vq::State::observe, "Get expectation of a given observable")
        .def("synchronize", &vq::State::synchronize, "Wait for completion of computation");

    auto finalize = [](void *ptr) noexcept { vq::finalize();};
    m.attr("_cleanup") = nb::capsule(reinterpret_cast<void*>(+finalize), finalize);

    vq::initialize();
}
