#!/bin/bash

##############################################################################################
## This is a template jobscript, designed to be run with the generate_jobscripts.sh script. ##
## DO NOT SUBMIT IT DIRECTLY, IT WON'T WORK.                                                ##
## generate_jobscripts.sh will subsitute 1 with the required number of MPI tasks,        ##
## saving you from modifying each of these scripts individually.                            ##
##############################################################################################

#SBATCH --job-name=image_4mpi
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=chem036964

# Ensure that the MPI module is loaded
module add openmpi/5.0.3-et6p

# Submit
srun --mpi=pmix_v2 ./test 1000 1 42 0.6 0