#!/bin/bash
#SBATCH --job-name=r2_finish
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/r2_systematic/logs/finish.log
#SBATCH --error=outputs/r2_systematic/logs/finish.log
cd /home/yanivpev/MentalHealthSympAI
bash run_r2_finish.sh 19405517 19405519
