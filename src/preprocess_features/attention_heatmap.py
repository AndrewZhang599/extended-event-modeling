import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv
from joblib import Parallel, delayed
from pandas import DataFrame
import itertools

# This script processes eye-tracking data for multiple movies to produce saliency maps
# and exports the results to CSV files. This is used to produce the analysis presented
# in the manuscript.

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
    return heatmap

def get_mean(array, xindex, yindex):
    """
    Calculate the mean of a subarray defined by grid indices.
    """
    temp = np.hsplit(np.vsplit(array, 6)[yindex], 8)[xindex]
    return np.mean(temp)

def process_movie(movie_file, output_csv):
    """
    Process eye-tracking data for a single movie and save results to a CSV file.
    """
    file_read = pd.read_csv("../data/all_eye.csv")
    movie_data = file_read[file_read["video"] == movie_file]
    frame_read = pd.read_csv(f"movie_{movie_file.replace('.', '')}_hand_grid_new.csv")
    frame_list = frame_read['frame'].tolist()
    unique_list = np.unique(frame_list).tolist()

    def save_saliency(f):
        all_value = []
        frame = movie_data[movie_data["frame"] == f]
        new_frame = frame.groupby(['sub']).mean()
        values = new_frame.to_numpy()
        all_temp = Fixpos2Densemap(values, 1280, 720)
        for i, j in itertools.product(range(8), range(6)):
            temp = get_mean(all_temp, i, j)
            temp_data = [i + 1, j + 1, temp]
            all_value.append(temp_data)
        return all_value

    results = Parallel(n_jobs=200)(delayed(save_saliency)(f) for f in unique_list)
    results = np.array(results)
    df = pd.DataFrame(results.reshape(-1, 3), columns=["gridx", "gridy", "hand_value"])
    df["frame"] = [element for element in unique_list for _ in range(48)]
    df.to_csv(output_csv, index=False)

# List of movies and corresponding output files
movies = ["1.2.3.mp4", "3.1.3.mp4", "2.4.1.mp4", "6.3.9.mp4"]
output_files = ["movie_123_gaze_grid.csv", "movie_313_gaze_grid.csv", "movie_241_gaze_grid.csv", "movie_639_gaze_grid.csv"]

# Process each movie
for movie, output in zip(movies, output_files):
    process_movie(movie, output)