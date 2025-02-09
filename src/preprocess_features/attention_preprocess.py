import pandas as pd
import numpy as np
import dask.dataframe as dd 
from pandas import DataFrame
from src.utils import parse_config, logger
from joblib import Parallel, delayed
import os 
import csv 

def GaussianMask(sizex, sizey, sigma=10, center=None, fix=1):
    """
    Generate a Gaussian mask.
    sizex  : mask width
    sizey  : mask height
    sigma  : Gaussian standard deviation
    center : Gaussian mean (center of the mask)
    fix    : Maximum value of the Gaussian function
    return : Gaussian mask as a 2D numpy array
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if not np.isnan(center[0]) and not np.isnan(center[1]):
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))
    return fix * np.exp(-((x - x0) ** 2 / sigma ** 2 + (y - y0) ** 2 / sigma ** 2))

def Fixpos2Densemap(fix_arr, width, height):
    """
    Convert fixation positions to a density map.
    fix_arr : array of fixation data [number of subjects x 3(x,y,fixation)]
    width   : output image width
    height  : output image height
    return  : heatmap as a 2D numpy array
    """
    heatmap = np.zeros((height, width), np.float32)
    for n_subject in range(fix_arr.shape[0]):
        center = (fix_arr[n_subject, 1], fix_arr[n_subject, 2])
        place = GaussianMask(width, height, 33, center=center)
        heatmap += place
    heatmap /= fix_arr.shape[0]
    normalized_heatmap = heatmap / np.sum(heatmap) #normalize the heatmap to sum to 1 
    return normalized_heatmap*100000 #scale values by 1000


def attention_weights_calculation(heatmap, tracking_df, frame):
    current_frame = tracking_df[tracking_df['frame'] == frame]
    attention_weights = {}
    epsilon = 1e-6 #lowest value is epsilon
    boxes = current_frame[['name', 'x', 'y', 'w', 'h']].values
    for row in boxes: 
        y = int(row[2])
        x = int(row[1])
        w = int(row[3]) 
        h = int(row[4])
        bounding_area = heatmap[y:y+h, x:x+w] 
        if np.sum(bounding_area) == 0: #make sure sum is not 0; if so, add epsilon 
            calc_attention = epsilon
        else: 
            calc_attention = np.mean(bounding_area) + epsilon   #make sure lowest value is epsilon       
        
        if np.isnan(calc_attention): #make sure not nan; rare but happens 
            attention_weights[row[0]] = epsilon
            
        else: 
            attention_weights[row[0]] = calc_attention
   
    # print(attention_weights.values())
    return frame, attention_weights


def generate_heatmaps_and_attention_weights(frame, movie_data, width, height, tracking_df): 
    current_frame = movie_data[movie_data['frame'] == frame]
    new_frame = current_frame.groupby(['sub']).mean()
    values = new_frame.to_numpy() 
    heatmap = Fixpos2Densemap(values, width, height)
    frame, attention_weights = attention_weights_calculation(heatmap, tracking_df, frame)
    return frame, heatmap, attention_weights


if __name__ == "__main__":  
    args = parse_config()
    logger.info(f"Running attention preprocess with the following config: {args}")
    tracking_df = pd.read_csv(args.tracking_path)
    eye_data = dd.read_csv(args.eye_data_path).compute() 
    output_dir = args.output_dir
    # width = args.width.astype(int)
    # height = args.height.astype(int)
    movie = args.video_name

    if not os.path.exists(os.path.join(output_dir, "attention_weights")):
        os.makedirs(os.path.join(output_dir, "attention_weights"), exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, "heatmaps")):
        os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)

    #save the attention weights and heatmaps as csv and npy files 
    attention_weights_path = os.path.join(output_dir, "attention_weights", f"{movie}_C1_attention_weights.csv")
    heatmaps_path = os.path.join(output_dir, "heatmaps", f"{movie}_C1_heatmaps.npy")  # adjust path as needed
    movie_data = eye_data[eye_data['video'] == f'{movie}.mp4']
    tracking_df['frame'] = tracking_df['frame'] + 1 #add one to the frame number to match the eye data
    unique_frames = np.sort(tracking_df['frame'].unique())
    # unique_frames = range(1,200)
    batch_size = 500 #process 500 frames at a time

    def process_frame(frame): 
        frame, heatmap, attention_weights = generate_heatmaps_and_attention_weights(
            frame, movie_data, 1280, 720, tracking_df
        )
        return frame, heatmap, attention_weights

    num_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    logger.info(f"Parallel processing with {num_jobs} jobs")

    with open(attention_weights_path, 'w', newline='') as f: 
        writer = csv.writer(f) 
        writer.writerow(['frame','object','attention_weight'])


        for i in range(0, len(unique_frames), batch_size):
            batch_frames = unique_frames[i:i+batch_size] 
            logger.info(f"Processing batch from {batch_frames[0]} to {batch_frames[-1]}")

            results = Parallel(n_jobs=num_jobs, verbose=10)(
                delayed(process_frame)(frame) for frame in batch_frames 
            )
            for frame, heatmap, attention_weights in results: 
                for obj, wt in attention_weights.items(): 
                    writer.writerow([frame, obj, wt]) 
            
            # for result in results: 
            #     frame, heatmap, attention_weights = result
            #     heatmaps[frame] = heatmap
            #     attention_weights_dict[frame] = attention_weights

            f.flush() 
            logger.info(f"Finished batch {i} to {i+batch_size}")
            # np.save(heatmaps_path, heatmaps)
    logger.info("All batches processed")
