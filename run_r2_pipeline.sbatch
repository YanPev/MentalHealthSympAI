#!/bin/bash
# Standalone SLURM job that runs the autonomous R2 downstream driver. Being its
# own job (separate cgroup) it survives client disconnection AND the agent
# session ending. Waits for the judge, then selects R2, trains, analyses, reports,
# commits. Usage: sbatch run_r2_pipeline.sbatch
#SBATCH --job-name=r2_driver
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/r2_systematic/logs/pipeline.log
#SBATCH --error=outputs/r2_systematic/logs/pipeline.log

cd /home/yanivpev/MentalHealthSympAI
bash run_r2_pipeline.sh 19395515 19395516
