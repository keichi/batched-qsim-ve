#!/bin/bash
#PBS -q sxs
#PBS --venode 1
#PBS -S /bin/bash
#PBS -l elapstim_req=01:00:00

export OMP_PROC_BIND=true

unset VE_PROGINF

[[ $PBS_O_WORKDIR ]] && cd $PBS_O_WORKDIR

samples=100000
batch_size=50000

export VE_LD_LIBRARY_PATH=$(realpath ../build):$VE_LD_LIBRARY_PATH

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0 --output-probs --trials 1 > random_cirtcuit_pdf_ideal.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.001 --output-probs --trials 1 > random_cirtcuit_pdf_1e-3.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.002 --output-probs --trials 1 > random_cirtcuit_pdf_2e-3.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.005 --output-probs --trials 1 > random_cirtcuit_pdf_5e-3.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.01 --output-probs --trials 1 > random_cirtcuit_pdf_1e-2.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.02 --output-probs --trials 1 > random_cirtcuit_pdf_2e-2.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.05 --output-probs --trials 1 > random_cirtcuit_pdf_5e-2.csv

../build/veqsim-random-circuit --samples $batch_size --batch-size $batch_size \
                               --noise-rate 0.1 --output-probs --trials 1 > random_cirtcuit_pdf_1e-1.csv
