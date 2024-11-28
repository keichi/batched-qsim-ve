#pragma once

#include <bit>
#include <cassert>
#include <complex>
#include <vector>

#include "state.hpp"

enum PauliID : std::uint64_t { I, X, Y, Z };

class PauliOperator
{
    friend class Observable;

public:
    PauliOperator(std::complex<double> coef, std::vector<std::uint64_t> targets,
                  std::vector<std::uint64_t> pauli_ids)
        : coef_(coef), targets_(targets), pauli_ids_(pauli_ids), bit_flip_mask_(0),
          phase_flip_mask_(0)
    {
        assert(targets.size() == pauli_ids.size());

        for (int i = 0; i < targets.size(); i++) {
            std::uint64_t target_qubit = targets[i];
            std::uint64_t pauli_id = pauli_ids[i];

            if (pauli_id == PauliID::X || pauli_id == PauliID::Y) {
                bit_flip_mask_ |= 1ULL << target_qubit;
            }
            if (pauli_id == PauliID::Y || pauli_id == PauliID::Z) {
                phase_flip_mask_ |= 1ULL << target_qubit;
            }
        }
    }

private:
    std::complex<double> coef_;
    std::vector<std::uint64_t> targets_;
    std::vector<std::uint64_t> pauli_ids_;
    std::uint64_t bit_flip_mask_;
    std::uint64_t phase_flip_mask_;
};

std::uint64_t insert_zero_to_basis_index(std::uint64_t basis_index, std::uint64_t insert_index)
{
    std::uint64_t mask = (1ULL << insert_index) - 1;
    std::uint64_t temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

std::array<std::complex<double>, 4> PHASE_90ROT()
{
    return {std::complex<double>(1, 0), std::complex<double>(0, 1), std::complex<double>(-1, 0),
            std::complex<double>(0, -1)};
}

class Observable
{
public:
    Observable() {}

    void add_operator(const PauliOperator &pauli) { terms_.push_back(pauli); }

    std::vector<std::complex<double>> get_expectation(const State &state) const
    {
        std::uint64_t nterms = terms_.size();
        std::uint64_t dim = state.dim();
        std::uint64_t batch_size = state.batch_size();
        std::vector<std::complex<double>> res(batch_size);

        for (std::uint64_t term_id = 0; term_id < nterms; term_id++) {
            std::uint64_t bit_flip_mask = terms_[term_id].bit_flip_mask_;
            std::uint64_t phase_flip_mask = terms_[term_id].phase_flip_mask_;
            std::complex<double> coef = terms_[term_id].coef_;

            if (bit_flip_mask == 0) {
                for (std::uint64_t idx = 0; idx < dim >> 1; idx++) {
                    std::uint64_t idx1 = idx << 1;
                    std::uint64_t idx2 = idx1 | 1;

                    for (std::uint64_t sample = 0; sample < batch_size; sample++) {
                        double tmp1 = (std::conj(state.amplitude(sample, idx1)) *
                                       state.amplitude(sample, idx1))
                                          .real();
                        if (std::popcount(idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        double tmp2 = (std::conj(state.amplitude(sample, idx2)) *
                                       state.amplitude(sample, idx2))
                                          .real();
                        if (std::popcount(idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res[sample] += coef * (tmp1 + tmp2);
                    }
                }
            } else {
                for (std::uint64_t idx = 0; idx < dim >> 1; idx++) {
                    std::uint64_t pivot =
                        sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
                    std::uint64_t global_phase_90rot_count =
                        std::popcount(bit_flip_mask & phase_flip_mask);
                    std::complex<double> global_phase = PHASE_90ROT()[global_phase_90rot_count % 4];
                    std::uint64_t basis_0 = insert_zero_to_basis_index(idx, pivot);
                    std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                    for (std::uint64_t sample = 0; sample < batch_size; sample++) {
                        double tmp = std::real(state.amplitude(sample, basis_0) *
                                               std::conj(state.amplitude(sample, basis_1)) *
                                               global_phase * 2.);
                        if (std::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                        res[sample] += coef * tmp;
                    }
                }
            }
        }

        return res;
    }

private:
    std::vector<PauliOperator> terms_;
};
