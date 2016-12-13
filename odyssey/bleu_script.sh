#!/bin/bash
#SBATCH -n 8                    # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 0-12:00              # Runtime in D-HH:MM
#SBATCH -p serial_requeue       # Partition to submit to
#SBATCH --mem=16000             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o hostname_%j.out      # File to which STDOUT will be written
#SBATCH -e hostname_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=philliphuang@college.harvard.edu # Email to which notifications will be sent
 
python bleu.py