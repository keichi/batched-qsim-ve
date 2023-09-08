#!/bin/bash
#PBS -q sxs
#PBS --venode 1
#PBS -S /bin/bash
#PBS -l elapstim_req=01:00:00

export OMP_PROC_BIND=true

unset VE_PROGINF

cd $PBS_O_WORKDIR

echo -e "qubits\tsamples\tbatch_size\truntime [s]"

samples=100000

for qubits in $(seq 8 14)
    do
    for batch_size in 1000 2000 5000 10000 20000 50000 100000
    do
        echo -n -e "$qubits\t$samples\t$batch_size\t"
        ../build/qsim-gate-bench --qubits $qubits --samples $samples --batch-size $batch_size
    done
done
