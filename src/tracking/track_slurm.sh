#!/bin/bash

#SBATCH --export=run=1.2.3_C1,tag=default
#SBATCH -p tier1_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --mincpus=8
#SBATCH --output=logs/%j.tracking.%x.out
#SBATCH --account=jeffrey_zacks

echo $run
echo $tag
#echo "track_v100.sh"
cd /home/a.j.zhang/extended-event-modeling/
source activate pt-37
python src/tracking/tracking_to_correct_label.py -c configs/config_tracking_to_correct_label.ini --run $run --track_tag $tag 2>&1 | tee "logs/$run$tag.log"
