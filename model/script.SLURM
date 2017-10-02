#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run MPI applications on 
# TACC's Maverick system.
#
# $Id: job.mpi 1580 2013-01-08 04:10:50Z karl $
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


