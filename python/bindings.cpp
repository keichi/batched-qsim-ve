#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include "../src/state.hpp"

namespace nb = nanobind;

NB_MODULE(veqsim, m) {
    nb::class_<State>(m, "State")
        .def(nb::init<UINT, UINT>())
        .def_prop_ro("dim", &State::dim)
        .def_prop_ro("batch_size", &State::batch_size)
        .def("amplitude", &State::amplitude)
        .def("re", &State::re)
        .def("im", &State::im)
        .def("get_probability", &State::get_probability)
        .def("set_zero_state", &State::set_zero_state)
        .def("act_x_gate", &State::act_x_gate)
        .def("act_y_gate", &State::act_y_gate)
        .def("act_z_gate", &State::act_z_gate)
        .def("act_h_gate", &State::act_h_gate)
        .def("act_t_gate", &State::act_t_gate)
        .def("act_rx_gate", &State::act_rx_gate)
        .def("act_ry_gate", &State::act_ry_gate)
        .def("act_rz_gate", &State::act_rz_gate)
        .def("act_sx_gate", &State::act_sx_gate)
        .def("act_sy_gate", &State::act_sy_gate)
        .def("act_sw_gate", &State::act_sw_gate)
        .def("act_cnot_gate", &State::act_cnot_gate)
        .def("act_cx_gate", &State::act_cx_gate)
        .def("act_cz_gate", &State::act_cz_gate)
        .def("act_iswaplike_gate", &State::act_iswaplike_gate)
        .def("act_depolarizing_gate_1q", &State::act_depolarizing_gate_1q)
        .def("act_depolarizing_gate_2q", &State::act_depolarizing_gate_2q)
        .def("synchronize", &State::synchronize);
}
