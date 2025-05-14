#pragma once

#include <cassert>
#include <complex>
#include <vector>

enum PauliID : std::uint32_t { I, X, Y, Z };

struct PauliOperator
{
    PauliOperator(std::complex<double> coef, const std::vector<std::uint32_t> &targets,
                  const std::vector<std::uint32_t> &pauli_ids)
        : coef(coef), targets(targets), pauli_ids(pauli_ids), bit_flip_mask(0),
          phase_flip_mask(0)
    {
        assert(targets.size() == pauli_ids.size());

        for (int i = 0; i < targets.size(); i++) {
            std::uint32_t target_qubit = targets[i];
            std::uint32_t pauli_id = pauli_ids[i];

            if (pauli_id == PauliID::X || pauli_id == PauliID::Y) {
                bit_flip_mask |= 1ULL << target_qubit;
            }
            if (pauli_id == PauliID::Y || pauli_id == PauliID::Z) {
                phase_flip_mask |= 1ULL << target_qubit;
            }
        }
    }

    std::complex<double> coef;
    std::vector<std::uint32_t> targets;
    std::vector<std::uint32_t> pauli_ids;
    std::uint32_t bit_flip_mask;
    std::uint32_t phase_flip_mask;
};

struct Observable
{
    Observable() {}

    void add_operator(const PauliOperator &pauli) { terms.push_back(pauli); }

    std::vector<PauliOperator> terms;
};
