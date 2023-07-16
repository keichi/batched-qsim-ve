#pragma once

#include <vector>

using UINT = unsigned int;
using ITYPE = unsigned long long;

void apply_single_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                             UINT BATCH_SIZE, UINT n, const double matrix_re[2][2],
                             const double matrix_im[2][2], UINT target);

void apply_two_qubit_gate(std::vector<double> &state_re, std::vector<double> &state_im,
                          UINT BATCH_SIZE, UINT n, const double matrix_re[4][4],
                          const double matrix_im[4][4], UINT target);

void apply_h_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target);

void apply_rx_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, double angle, UINT target);

void apply_cnot_gate(std::vector<double> &state_re, std::vector<double> &state_im, UINT BATCH_SIZE,
                   UINT n, UINT target, UINT control);
