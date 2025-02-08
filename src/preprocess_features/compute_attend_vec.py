#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
from src.utils import logger, parse_config 


def process_frame(frame_df, glove_vectors, emb_dim):
    """
    Process a single frame (group of rows) and compute its weighted average semantic vector.
    Each row corresponds to an object with its associated attention weight.
    
    Parameters:
      frame_df (pd.DataFrame): DataFrame for one frame with columns 'object' and 'attention_weight'
      glove_vectors (dict): Mapping from words to their GloVe vectors (numpy arrays)
      emb_dim (int): Dimensionality of the embedding vectors
      
    Returns:
      np.ndarray: 1D numpy array of length emb_dim representing the frame's averaged vector.
                  Returns an array of NaNs if no valid words are found.
    """
    frame_sum = np.zeros((1, emb_dim))
    count = 0
    for _, row in frame_df.iterrows():
        # Split the object string into words and remove non-alphabetical characters.
        raw_words = row['object'].split(' ')
        words = [''.join(ch for ch in word if ch.isalpha()) for word in raw_words if word.strip() != '']
        if not words:
            continue
       #averages the vectors for multi-word objects 
        vec = np.zeros((1, emb_dim))
        valid_word_count = 0
        for word in words:
            try:
                vec += glove_vectors[word]
                valid_word_count += 1
            except KeyError:
                continue
        if valid_word_count == 0:
            continue
        vec /= valid_word_count
        # Multiply the averaged vector by its attention weight.
        weighted_vec = vec * row['attention_weight']
        frame_sum += weighted_vec
        count += 1
    if count == 0:
        return np.full((emb_dim,), np.nan)
    return (frame_sum / count).flatten()



if __name__ == '__main__':
    args = parse_config()
    attention_csv_path = os.path.join(args.attention_weights, f"{args.video_name}_attention_weights.csv")
    glove_pickle_path = args.glove_pickle  
    output_csv_path = os.path.join(args.output_dir, f"{args.video_name}_scene_vectors.csv")
    
    logger.info(f"Processing video: {args.video_name}")
    logger.info(f"Using attention weights from: {attention_csv_path}")
   
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True) 

    try:
        one_attention_weights = pd.read_csv(attention_csv_path)
    except Exception as e:
        sys.exit(f"Error reading {attention_csv_path}: {e}")
    
    # Verify that required columns are present.
    required_cols = {'frame', 'object', 'attention_weight'}
    if not required_cols.issubset(one_attention_weights.columns):
        sys.exit(f"Input CSV must contain columns: {required_cols}")
    
    try:
        with open(glove_pickle_path, 'rb') as f:
            glove_vectors = pkl.load(f)
    except Exception as e:
        sys.exit(f"Error loading GloVe vectors from {glove_pickle_path}: {e}")
  
    try:
        emb_dim = glove_vectors['apple'].shape[0]
    except Exception as e:
        sys.exit(f"Error determining embedding dimension: {e}")
    
    # Group the attention weights by frame.
    grouped = one_attention_weights.groupby('frame')
    results = []
    for frame, group in grouped:
        avg_vec = process_frame(group, glove_vectors, emb_dim)
        if frame % 1000 == 0: 
            logger.info(f"Processed frame {frame}") 
        results.append((frame, avg_vec))
    
    # Sort results by frame number.
    results.sort(key=lambda x: x[0])
    
    # Build a DataFrame where each row is a frame and each column is one dimension of the scene vector.
    out_dict = {'frame': [frame for frame, _ in results]}
    for i in range(emb_dim):
        col_name = f'attention_{i+1}'
        out_dict[col_name] = [vec[i] for _, vec in results]
    out_df = pd.DataFrame(out_dict)
    
    out_df.to_csv(output_csv_path, index=False) 
    logger.info(f"Saved scene vectors to {output_csv_path}") 


