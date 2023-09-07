#!/bin/bash

for noise_rate in 0.001 0.01 0.1
    do
    for batch_size in 1000 2000 5000 10000 20000 50000 100000
    do
        ./qsim-random-circuit --samples $batch_size --batch-size $batch_size --trials 1 \
                              --noise-rate $noise_rate
    done
done
