#!/bin/bash
#PBS -q sxs
#PBS --venode 1
#PBS -S /bin/bash
#PBS -l elapstim_req=06:00:00

export OMP_PROC_BIND=true

unset VE_PROGINF

[[ $PBS_O_WORKDIR ]] && cd $PBS_O_WORKDIR

echo -e "noise_rate\tsamples\tbatch_size\truntime [s]"

samples=100000

export VE_LD_LIBRARY_PATH=$(realpath ../build):$VE_LD_LIBRARY_PATH

for noise_rate in 0.001 0.002 0.005 0.01 0.02 0.05 0.1
    do
    for batch_size in 1 10 100 200 500 1000 2000 5000 10000 20000 50000
    do
        echo -n -e "$noise_rate\t$samples\t$batch_size\t"
        ../build/veqsim-random-circuit --samples $samples --batch-size $batch_size \
                                       --width 4 --height 4 --noise-rate $noise_rate --trials 1
    done
done
