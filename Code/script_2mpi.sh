#!/bin/bash

##############################################################################################
## This is a template jobscript, designed to be run with the generate_jobscripts.sh script. ##
## DO NOT SUBMIT IT DIRECTLY, IT WON'T WORK.                                                ##
## generate_jobscripts.sh will subsitute 1 with the required number of MPI tasks,        ##
## saving you from modifying each of these scripts individually.                            ##
##############################################################################################

#SBATCH --job-name=image_2mpi
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=chem036964

# Ensure that the MPI module is loaded
module add openmpi/5.0.3-et6p

# Submit
srun --mpi=pmix_v2 ./test "input_grid.txt" 1