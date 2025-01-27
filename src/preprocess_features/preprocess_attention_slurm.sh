#!/bin/bash


#SBATCH -p tier2_cpu
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.attention.%x.out
#SBATCH --account=jeffrey_zacks

#echo "track_v100.sh"
cd /home/a.j.zhang/extended-event-modeling/
source activate sem-viz-jupyter
python src/preprocess_features/attention_preprocess.py -c configs/config_preprocess_attention.ini"
