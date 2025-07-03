#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../src/state.hpp"

namespace nb = nanobind;
namespace vq = veqsim;

using namespace nb::literals;

NB_MODULE(_veqsim, m)
{
    nb::enum_<vq::PauliID>(m, "PauliID")
        .value("I", vq::PauliID::I)
        .value("X", vq::PauliID::X)
        .value("Y", vq::PauliID::Y)
        .value("Z", vq::PauliID::Z);

    nb::class_<vq::PauliOperator>(m, "PauliOperator")
        .def(nb::init<std::complex<double>, std::vector<UINT>, std::vector<UINT>>(),
             "coef"_a, "targets"_a, "pauli_ids"_a);

    nb::class_<vq::Observable>(m, "Observable")
        .def(nb::init<>())
        .def("add_operator", &vq::Observable::add_operator, "operator"_a,
             "Add a Pauli operator to the observable");

    nb::class_<vq::State>(m, "State", "A batched state vector")
        .def(nb::init<UINT, UINT>(), "num_qubits"_a, "num_samples"_a, "Create state vector")
        .def_prop_ro("dim", &vq::State::dim, "Dimension")
        .def_prop_ro("batch_size", &vq::State::batch_size, "Batch size")
        .def("get_vector", &vq::State::get_vector, "sample"_a, "Get a single state vector")
        .def("amplitude", &vq::State::amplitude, "sample"_a, "basis"_a,
             "Get the amplitude of a basis")
        .def("re", &vq::State::re, "sample"_a, "basis"_a,
             "Get the real component of the amplitude of a basis")
        .def("im", &vq::State::im, "sample"_a, "basis"_a
             "Get the imaginary component of the amplitude of a basis")
        .def("get_probability", nb::overload_cast<UINT>(&vq::State::get_probability, nb::const_),
             "basis"_a, "Get the probability of a basis to be observed")
        .def("get_probability", nb::overload_cast<UINT, UINT>(&vq::State::get_probability, nb::const_),
             "sample"_a, "basis"_a, "Get the probability of a basis to be observed")
        .def("set_zero_state", &vq::State::set_zero_state, "Initialize to |0..0>")
        .def("act_x_gate", &vq::State::act_x_gate, "target"_a, "Apply an X gate")
        .def("act_y_gate", &vq::State::act_y_gate, "target"_a, "Apply a Y gate")
        .def("act_z_gate", &vq::State::act_z_gate, "target"_a, "Apply a Z gate")
        .def("act_h_gate", &vq::State::act_h_gate, "target"_a, "Apply an H gate")
        .def("act_t_gate", &vq::State::act_t_gate, "target"_a, "Apply a T gate")
        .def("act_rx_gate", nb::overload_cast<UINT, double>(&vq::State::act_rx_gate),
             "target"_a, "theta"_a, "Act an RX gate with the same angle")
        .def("act_rx_gate",
             nb::overload_cast<UINT, const std::vector<double> &>(&vq::State::act_rx_gate),
             "target"_a, "theta"_a, "Act an RX gate with different angles")
        .def("act_ry_gate", nb::overload_cast<UINT, double>(&vq::State::act_ry_gate),
             "target"_a, "theta"_a, "Apply an RY gate")
        .def("act_ry_gate",
             nb::overload_cast<UINT, const std::vector<double> &>(&vq::State::act_ry_gate),
             "target"_a, "theta"_a, "Act an RY gate with different angles")
        .def("act_rz_gate", nb::overload_cast<UINT, double>(&vq::State::act_rz_gate),
             "target"_a, "theta"_a, "Apply an RZ gate")
        .def("act_rz_gate",
             nb::overload_cast<UINT, const std::vector<double> &>(&vq::State::act_rz_gate),
             "target"_a, "theta"_a, "Act an RZ gate with different angles")
        .def("act_p_gate", nb::overload_cast<UINT, double>(&vq::State::act_p_gate),
             "target"_a, "theta"_a, "Apply a P gate")
        .def("act_p_gate",
             nb::overload_cast<UINT, const std::vector<double> &>(&vq::State::act_p_gate),
             "target"_a, "theta"_a, "Act a P gate with different angles")
        .def("act_sx_gate", &vq::State::act_sx_gate,  "target"_a, "Apply an SX gate")
        .def("act_sy_gate", &vq::State::act_sy_gate, "target"_a, "Apply an SY gate")
        .def("act_sw_gate", &vq::State::act_sw_gate, "target"_a, "Apply an SW gate")
        .def("act_cnot_gate", &vq::State::act_cnot_gate, "control"_a, "target"_a, "Apply a CNOT gate")
        .def("act_cx_gate", &vq::State::act_cx_gate, "control"_a, "target"_a, "Apply a CX gate")
        .def("act_cz_gate", &vq::State::act_cz_gate, "control"_a, "target"_a, "Apply a CZ gate")
        .def("act_iswaplike_gate", &vq::State::act_iswaplike_gate, "control"_a, "target"_a,
              "theta"_a, "Apply an iSWAP-like gate")
        .def("act_depolarizing_gate_1q", &vq::State::act_depolarizing_gate_1q,
             "target"_a, "probability"_a, "Apply an one-qubit depolarizing noise gate")
        .def("act_depolarizing_gate_2q", &vq::State::act_depolarizing_gate_2q,
             "control"_a, "target"_a, "probability"_a, "Apply a two-qubit depolarizing noise gate")
        .def("observe", &vq::State::observe, "observable"_a, "Get expectation of a given observable")
        .def("synchronize", &vq::State::synchronize, "Wait for completion of computation");

    auto finalize = [](void *ptr) noexcept { vq::finalize();};
    m.attr("_cleanup") = nb::capsule(reinterpret_cast<void*>(+finalize), finalize);

    vq::initialize();
}
