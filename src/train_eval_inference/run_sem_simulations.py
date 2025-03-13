import subprocess
import datetime
import os
import glob

# hand weighted
# trigger = 'pe'
# thresholds = [5e-1]  #originally 4.3e-1
# alfa = [2e4] #originally 2e-2
# lmda = [5e-4] #originally 5e0

trigger = 'uncertainty'
thresholds = [4e-3] #originally 4.6e-3
lmda = [5e-5] #originally 5e-5
alfa = [5e4] #originally 1e6

# #attention weighted
# trigger = 'pe'
# thresholds = [5e-1,5.3e-1]  #originally 4.3e-1
# alfa = [2e4, 2e5] #originally 2e-2
# lmda = [5e-4,5e-5] #originally 5e0

# trigger = 'uncertainty'
# thresholds = [6.0e-3] #originally 2.5e-3
# lmda = [5e-4] #originally 5e-3
# alfa = [1e5] #originally 1e0


# Define validation sets - we want to run simulations for all permutations of each val set
validation_ids = ['6.3.9', '3.1.3', '2.4.1', '1.2.3']

active_val_id = '6.3.9'
# active_val_id = '3.1.3'
# active_val_id = '2.4.1'
# active_val_id = '1.2.3'


output_dir = "output"
train_pattern = os.path.join(output_dir, f"trainval{active_val_id}_permutation_*.txt")
train_files = sorted(glob.glob(train_pattern))
val_file = os.path.join(output_dir, f"val{active_val_id}.txt")

# # train_file = 'output/train_val3.1.3.txt'  #train on everything but the last number
# # val_file = 'output/val3.1.3.txt'
# #
# train_file = 'output/train_val2.4.1.txt'  #train on everything but the last number
# val_file = 'output/val2.4.1.txt'
# #
# # train_file = 'output/train_val1.2.3.txt'  #train on everything but the last number
# # val_file = 'output/val1.2.3.txt'

print(f"Found {len(train_files)} permutation files for validation ID {active_val_id}")
for tf in train_files:
    print(f"  - {os.path.basename(tf)}")

nodes = list(range(4,6))
# nodes = list(range(2,6)) + list(range(8,10))
# nodes = list(range(10,16))
# nodes = list(range(17,22)) + list(range(24,25))
# nodes = list(range(25,26)) + list(range(27,32))
equal_sigma = 0  
ver_sem = 0 #0 for hand, 1 for attention
node_index = 0
seed = 1111
current_date = 'mar_12'
# nodes = list(range(2, 7)) + list(range(8, 15)) + list(range(16, 23)) + list((24,25))+list(range(27,33))


if trigger == 'pe' and ver_sem == 0:
    sem_version = "pe_hand"
elif trigger == 'pe' and ver_sem == 1:
    sem_version = "pe_attention"
elif trigger == 'uncertainty' and ver_sem == 0:
    sem_version = "uncertainty_hand"
elif trigger == 'uncertainty' and ver_sem == 1:
    sem_version = "uncertainty_attention"
else:
    sem_version = "custom"

# for pe_threshold in pe_thresholds:
for lr in [1e-3]:
    for threshold in thresholds:
        for a in alfa:
            for l in lmda:
                # For each permutation of training data
                for perm_idx, train_file in enumerate(train_files, 1):
                    perm_name = os.path.basename(train_file).replace(f"trainval{active_val_id}_", "").replace(".txt", "")
                    
                    tag = f'{current_date}_{sem_version}_val{active_val_id}_{perm_name}_{threshold:.1E}_{a:.1E}_{l:.1E}'
                    
                    if node_index >= len(nodes):
                        print(f"Warning: Not enough nodes for all simulations. Stopping at {node_index}")
                        break
                        
                    selected_node = nodes[node_index]
                    
                    # Extract human-readable permutation name for display
                    val_data = active_val_id
                    
                    print(f"Running {sem_version} on validation data {val_data} with permutation {perm_name} on node{selected_node:02d}")

                    proc = subprocess.Popen(
                        ['sbatch', '--job-name', tag,
                         '--nodelist', f'node{selected_node:02d}',
                         # NOTE: hard-code here is dangerous, changing name of directory in local wouldn't refactor this, combined
                         # with not deleting in remote can result in running old code
                         'src/train_eval_inference/model_corpus_andrew.sh',
                         train_file, #$1
                         val_file, #$2
                         f'{a:.1E}', #$3
                         f'{l:.1E}', #$4
                         tag, #$5
                         trigger, #$6
                         f'{threshold:.1E}', #$7
                         f'{seed}',  #$8
                         f'{lr:.0E}', #$9
                         f'{equal_sigma}']) #$10
                    node_index += 1
