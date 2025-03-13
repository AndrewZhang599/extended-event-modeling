import subprocess

# trigger = 'pe'
# thresholds = [4.3e-1]
# alfa = [2e-2]  # for PE
# lmda = [5e0]  # for PE

# trigger = 'always'
# thresholds = [1e-1]  # just as a placeholder
# alfa = [1e-1]  # for always
# lmda = [5e5]  # for always

# hand weighted
trigger = 'pe'
thresholds = [5e-1]  #originally 4.3e-1
alfa = [2e4] #originally 2e-2
lmda = [5e-4] #originally 5e0

# trigger = 'uncertainty'
# thresholds = [4.6e-3,5e-3] #originally 2.5e-3
# lmda = [5e-3,5e-4,5e-5] #originally 5e-3
# alfa = [1e6] #originally 1e0

# #attention weighted
# trigger = 'pe'
# thresholds = [5e-1,5.3e-1]  #originally 4.3e-1
# alfa = [2e4, 2e5] #originally 2e-2
# lmda = [5e-4,5e-5] #originally 5e0

# trigger = 'uncertainty'
# thresholds = [6.3e-3] #originally 2.5e-3
# lmda = [5e-2,5e-1,5e0] #originally 5e-3
# alfa = [1e5] #originally 1e0

# nodes = list(range(2,7)) + list(range(24,26)) + list(range(27,28))
nodes = list(range(23,25)) #pe hand
seed = 1111
equal_sigma = 0
node_index = 0
# nodes = list(range(2, 7)) + list(range(8, 15)) + list(range(16, 23)) + list((24,25))+list(range(27,33))

# for pe_threshold in pe_thresholds:
for lr in [1e-3]:
    for threshold in thresholds:
        for a in alfa:
            for l in lmda:
                # Create tag with feb_26_uncertainty_hand as requested
                tag = f'feb_35_pe_hand_{threshold:.1E}_{a:.1E}_{l:.1E}'
                selected_node = nodes[node_index]

                proc = subprocess.Popen(
                    ['sbatch', '--job-name', tag,
                     '--nodelist', f'node{selected_node:02d}',
                     # NOTE: hard-code here is dangerous, changing name of directory in local wouldn't refactor this, combined
                     # with not deleting in remote can result in running old code
                     'src/train_eval_inference/model_corpus_andrew.sh',
                     # 'output/train_sep_09.txt', #$1
                     'output/train_sep_09_copy.txt', #$1
                     # 'output/valid_sep_09.txt', #$2
                     'output/train_sep_09_copy.txt', #$2
                     f'{a:.1E}', #$3
                     f'{l:.1E}', #$4
                     tag, #$5
                     trigger, #$6
                     f'{threshold:.1E}', #$7
                     f'{seed}',  #$8
                     f'{lr:.0E}', #$9
                     f'{equal_sigma}']) #$10
                node_index += 1
