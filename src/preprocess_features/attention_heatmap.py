import pandas as pd
import numpy as np
import dask.dataframe as dd 
from pandas import DataFrame
# from src.utils import parse_config
import os 

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
    return normalized_heatmap*1000 #scale values by 1000 


def generate_heatmaps_and_attention_weights(frame, movie_data, width, height, tracking_df): 
    current_frame = movie_data[movie_data['frame'] == frame]
    new_frame = current_frame.groupby(['sub']).mean()
    values = new_frame.to_numpy() 
    heatmap = Fixpos2Densemap(values, width, height)

    def attention_weights(heatmap, tracking_df, frame):
        current_frame = tracking_df[tracking_df['frame'] == frame]
        attention_weights = {}
        epsilon = 1e-6 #make sure the average is not 0 

        boxes = current_frame[['name', 'x', 'y', 'w', 'h']].values
        attention_weights = {
            row[0]: np.mean(heatmap[int(row[2]):int(row[2])+int(row[4]), 
                                   int(row[1]):int(row[1])+int(row[3])]) + epsilon
            for row in boxes
        }
        
        total = sum(attention_weights.values())
        attention_weights = {k: v / total for k, v in attention_weights.items()} #normalize the attention weights
       
        print(attention_weights.values())
        return attention_weights
    attention_weights = attention_weights(heatmap, tracking_df, frame)
    return heatmap, attention_weights


if __name__ == "__main__":  
    # args = parse_config()
    tracking_df = pd.read_csv(r'C:\Users\super\OneDrive\Desktop\Github repos\extended-event-modeling\output\tracking_all\1.2.3_C1_r50.csv')
    eye_data = dd.read_csv(r'C:\Users\super\OneDrive\Desktop\Github repos\extended-event-modeling\output\eye_all\all_eye_080422.csv').compute() 
    output_dir = r"C:\Users\super\OneDrive\Desktop\Github repos\extended-event-modeling\output"  # adjust path as needed

    if not os.path.exists(os.path.join(output_dir, "attention_weights")):
        os.makedirs(os.path.join(output_dir, "attention_weights"), exist_ok=True)
    if not os.path.exists(os.path.join(output_dir, "heatmaps")):
        os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)

    #save the attention weights and heatmaps as csv and npy files 
    attention_weights_path = os.path.join(output_dir, "attention_weights", "1.2.3_C1_attention_weights.csv")
    heatmaps_path = os.path.join(output_dir, "heatmaps", "1.2.3_C1_heatmaps.npy")  # adjust path as needed
    movie_data = eye_data[eye_data['video'] == '1.2.3.mp4']
    # unique_frames = tracking_df.index.unique().sort_values() 
    unique_frames = range(1,10)

    heatmaps = dict()
    attention_weights_dict = dict()
    for f in unique_frames: 
        heatmap, attention_weights = generate_heatmaps_and_attention_weights(f, movie_data, 1280, 720, tracking_df)
        heatmaps[f] = heatmap 
        attention_weights_dict[f] = attention_weights

    attention_df = pd.DataFrame.from_dict(attention_weights_dict, orient='index')
    attention_df.to_csv(attention_weights_path)

    np.save(heatmaps_path, heatmaps)

