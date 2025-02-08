#!/bin/bash


#SBATCH -p tier2_cpu
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.attention.%x.out
#SBATCH --account=jeffrey_zacks

#echo "track_v100.sh"
cd /home/a.j.zhang/extended-event-modeling/
export PYTHONPATH="$PYTHONPATH:/home/a.j.zhang/extended-event-modeling/"
source activate sem-viz-jupyter-frozen
python src/preprocess_features/compute_attend_vec.py -c configs/config_compute_attend_vec.ini
