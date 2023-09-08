#!/bin/bash
#PBS -q sxs
#PBS --venode 1
#PBS -S /bin/bash
#PBS -l elapstim_req=01:00:00

export OMP_PROC_BIND=true

unset VE_PROGINF

[[ $PBS_O_WORKDIR ]] && cd $PBS_O_WORKDIR

echo -e "noise_rate\tsamples\tbatch_size\truntime [s]"

samples=100000

for noise_rate in 0.001 0.005 0.01 0.05 0.1
    do
    for batch_size in 1000 2000 5000 10000 20000 50000
    do
        echo -n -e "$noise_rate\t$samples\t$batch_size\t"
        ../build/qsim-random-circuit --samples $batch_size --batch-size $batch_size \
                                     --noise-rate $noise_rate --trials 1
    done
done
