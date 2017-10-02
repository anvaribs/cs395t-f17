#!/bin/bash
#----------------------------------------------------
# TACC's Maverick system.
#----------------------------------------------------

#SBATCH -J model_builder              # Job name
#SBATCH -o myjob.o%j          # Name of stdout output file (%j expands to jobId)
#SBATCH -e myjob.e%j 	      # Name of stderr error file
#SBATCH -p gpu                # Queue name
#SBATCH -N 1                  # Total number of nodes requested (20 cores/node)
#SBATCH -n 20                 # Total number of mpi tasks requested
#SBATCH -t 02:00:00           # Run time (hh:mm:ss) 
#SBATCH --mail-user=anvari@utexas.edu
#SBATCH --mail-type=all	      # Send email at the begin and end of job
#SBATCH -A cs395t_f17	      # Allocation name (req'd if you have more than 1)


# Other commands mus follow all #SBATCH directives ...

module list
pwd
date

# Launch the MPI executable named "a.out"

python3 run_experiments.py --conf_file='conf_test.csv'


