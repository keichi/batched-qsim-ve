#!/bin/bash
#PBS -q sxs
#PBS --venode 1
#PBS -S /bin/bash
#PBS -l elapstim_req=06:00:00

export OMP_PROC_BIND=true

unset VE_PROGINF

[[ $PBS_O_WORKDIR ]] && cd $PBS_O_WORKDIR

echo -e "noise_rate\tqubits\tsamples\tbatch_size\truntime [s]"

samples=100000

export VE_TRACEBACK=VERBOSE
export VE_LD_LIBRARY_PATH=$(realpath ../build):$VE_LD_LIBRARY_PATH

for noise_rate in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
do
    for qubits in $(seq 8 14)
        do
        for batch_size in 1 10 100 200 500 1000 2000 5000 10000 20000 50000 100000
        do
            echo -n -e "$noise_rate\t$qubits\t$samples\t$batch_size\t"
            ../build/qsim-gate-bench --qubits $qubits --samples $samples --batch-size $batch_size \
                                     --noise-rate $noise_rate --gate NOISE
        done
    done
done

for noise_rate in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
do
    for qubits in $(seq 15 25)
        do
        for batch_size in 100 200 500 1000 2000 5000 10000 20000 50000 100000
        do
            if [[ $((80 * 1000 * 1000 * 1000)) -lt $((16 * 2 ** $qubits * $batch_size)) ]]; then
                continue
            fi
            echo -n -e "$noise_rate\t$qubits\t$samples\t$batch_size\t"
            ../build/qsim-gate-bench --qubits $qubits --samples $samples --batch-size $batch_size \
                                     --noise-rate $noise_rate --gate NOISE --depth 1
        done
    done
done
