#!/bin/bash

export OMP_NUM_THREADS=1

echo -e "qubits\ttarget\truntime [s]"

for qubits in $(seq 14 2 20)
    do
    for target in $(seq 0 $(($qubits-1)))
    do
        echo -n -e "$qubits\t$target\t"
        build/qsim-bench --qubits $qubits --samples 1 --batch-size 1 \
                         --target $target --depth 100
    done
done
