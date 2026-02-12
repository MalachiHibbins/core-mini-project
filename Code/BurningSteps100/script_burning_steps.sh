#!/bin/bash

##############################################################################################
## This is a template jobscript, designed to be run with the generate_jobscripts.sh script. ##
## DO NOT SUBMIT IT DIRECTLY, IT WON'T WORK.                                                ##
## generate_jobscripts.sh will subsitute 1 with the required number of MPI tasks,        ##
## saving you from modifying each of these scripts individually.                            ##
##############################################################################################

#SBATCH --job-name=image_16mpi
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=chem036964

# Ensure that the MPI module is loaded
module add openmpi/5.0.3-et6p

# Submit: run the executable for a range of probabilities
# Loop probabilities from 0.05 to 0.95 in steps of 0.05
for i in $(seq 5 5 95); do
	p=$(awk -v i="$i" 'BEGIN{printf "%.2f", i/100}')
	echo "Running with probability ${p}"
	srun --mpi=pmix_v2 ./test 100 100 42 "${p}" 0
done
