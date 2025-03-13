#!/usr/bin/env python3
import os
import sys
import math
import re
import traceback
import pickle as pkl
from copy import deepcopy

import numpy as np
import pandas as pd

from src.utils import parse_config, logger, contain_substr


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(float)
    return appear_df


def preprocess_optical(vid_csv, standardize=True):
    vid_df = pd.read_csv(vid_csv, index_col='frame')
    for c in vid_df.columns:
        if not standardize:
            vid_df.loc[:, c] = (vid_df[c] - min(vid_df[c].dropna())) / (
                max(vid_df[c].dropna()) - min(vid_df[c].dropna()))
        else:
            vid_df.loc[:, c] = (vid_df[c] - vid_df[c].dropna().mean()) / vid_df[c].dropna().std()
    return vid_df


def preprocess_skel(skel_csv, use_position=0, standardize=True, feature_tag='',
                    ratio_features=0.8, ratio_samples=0.8, stats_skel_csv='') -> (pd.DataFrame, bool):
    skel_df = pd.read_csv(skel_csv, index_col='frame')
    skel_df.drop(['sync_time', 'raw_time', 'body', 'J1_dist_from_J1', 'J1_3D_rel_X', 'J1_3D_rel_Y', 'J1_3D_rel_Z'], axis=1,
                 inplace=True)
    if use_position:
        keeps = ['accel', 'speed', 'dist', 'interhand', '2D', 'rel']
    else:
        keeps = ['accel', 'speed', 'dist', 'interhand', 'rel']

    for c in skel_df.columns:
        if contain_substr(c, keeps):
            continue
        else:
            skel_df.drop([c], axis=1, inplace=True)

    defective = 0
    # load sampled skel features, 200 samples for each video.
    combined_runs = pd.read_csv(f'{stats_skel_csv}')
    # mask outliers with N/A
    select_indices = (skel_df < combined_runs.quantile(.95)) & (skel_df > combined_runs.quantile(.05))
    skel_df = skel_df[select_indices]
    qualified_columns = (select_indices.sum() > int(len(skel_df) * ratio_samples))
    if qualified_columns.sum() / len(qualified_columns) <= ratio_features:
        logger.info(f"Video {skel_csv} has {len(qualified_columns) - qualified_columns.sum()} "
                    f"un-qualified columns out of {len(qualified_columns)}!!!")
        defective = 1
        # open(filtered_txt, 'a').write(f"{skel_csv}\n")

    # fill N/A
    skel_df = skel_df.ffill()

    if standardize:
        # standardize using global statistics
        select_indices = (combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))
        combined_runs_q = combined_runs[select_indices]
        stats = combined_runs_q.describe().loc[['mean', 'std']]
        skel_df = (skel_df - stats.loc['mean', skel_df.columns]) / stats.loc['std', skel_df.columns]

    return skel_df, defective


def interpolate_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    first_frame = dataframe.index[0]
    last_frame = dataframe.index[-1]
    dummy_frame = pd.DataFrame(np.NaN, index=range(first_frame, last_frame),
                               columns=dataframe.columns)
    dummy_frame = dummy_frame.combine_first(dataframe).interpolate(limit_area='inside')
    return dummy_frame


# ----------------------------------------------------------------------------------
# New helper function from compute_attend_vec.py for attention-weighted embeddings
# ----------------------------------------------------------------------------------

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
        raw_words = row['object'].split(' ')
        words = [''.join(ch for ch in word if ch.isalpha()) for word in raw_words if word.strip() != '']
        if not words:
            continue
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
        weighted_vec = vec * row['attention_weight']
        frame_sum += weighted_vec
        count += 1
    if count == 0:
        return np.full((emb_dim,), np.nan)
    return (frame_sum / count).flatten()


class FeatureProcessor:
    """
    This class loads individual features and pre-processes them before feeding them to the network.
    In this version, the only scene feature comes from the attention–weighted semantic vectors.
    """
    def __init__(self, configs, glove_vectors=None, emb_dim=None):
        self.feature_tag = configs.feature_tag
        self.skel_csv_dir = configs.skel_csv
        self.ratio_samples = float(configs.ratio_samples)
        self.ratio_features = float(configs.ratio_features)
        self.filtered_txt = f"output/filtered_skel_{self.feature_tag}_{self.ratio_samples}_{self.ratio_features}.txt"
        self.use_skel_position = configs.use_skel_position
        self.optical_csv_dir = configs.optical_csv
        self.appear_csv_dir = configs.appear_csv
        self.out_preprocess_pkl = configs.out_preprocess_pkl
        self.stats_skel_csv = configs.stats_skel_csv
        self.run = configs.run
        # have a run to fps df
        self.run_specs_df = pd.read_csv(f'{configs.run_specs}')
        run = self.run + ('_kinect' if '_kinect' not in self.run else '')
        self.fps = float(self.run_specs_df.loc[self.run_specs_df.run == run, 'fps'].iloc[0])
        self.rate = configs.rate
        self.df_dict = dict()
        self.complete_txt = f"output/preprocessed_complete_{args.feature_tag}.txt"
        # This class process at a run level, list level processes should be in parallel_preprocess_indv_run.py
        # if os.path.exists(self.complete_txt):
        #     os.remove(self.complete_txt)
        self.error_txt = f"output/preprocessed_error_{args.feature_tag}.txt"
        # if os.path.exists(self.error_txt):
        #     os.remove(self.error_txt)
        self.video_name = configs.video_name
        self.attention_weights_dir = configs.attention_weights
        self.glove_vectors = glove_vectors
        self.emb_dim = emb_dim

    def resample_df(self, df) -> pd.DataFrame:
        out_df = df.set_index(pd.to_datetime(df.index / self.fps, unit='s'), drop=False, verify_integrity=True)
        # dummy_frame is necessary in case df has missing frames and needs interpolation
        # though, df was already interpolated before in combine_data_frames and even after -> TODO: remove dummy_frame
        resample_index = pd.date_range(start=out_df.index[0], end=out_df.index[-1], freq=self.rate)
        dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=out_df.columns)
        # resample().mean() will make frame_id not integer anymore
        out_df = out_df.combine_first(dummy_frame).interpolate(method='time', limit_area='inside').resample(self.rate).mean()
        return out_df

    def combine_dataframes(self, data_frames) -> pd.DataFrame:
        # Some features such as optical flow are calculated not for all frames, interpolate first
        data_frames = [interpolate_frame(df) for df in data_frames]
        combine_df = pd.concat(data_frames, axis=1)
        # After dropping null values, variances are not unit anymore, some are around 0.8.
        combine_df.dropna(axis=0, inplace=True)
        combine_df['frame'] = combine_df.index
        # After resampling, some variances drop to 0.3 or 0.4
        combine_df = self.resample_df(combine_df)
        # because resample use mean, need to adjust categorical variables
        combine_df['appear'] = combine_df['appear'].apply(math.ceil).astype(float)
        combine_df['disappear'] = combine_df['disappear'].apply(math.ceil).astype(float)

        assert combine_df.isna().sum().sum() == 0
        combine_df.set_index('frame', inplace=True)
        combine_df.drop(['sync_time', 'frame'], axis=1, inplace=True, errors='ignore')
        return combine_df

    def preprocess_appear_feature(self) -> pd.DataFrame:
        logger.info('Processing Appear features...')
        pd.set_option('use_inf_as_na', True)
        appear_csv = os.path.join(self.appear_csv_dir, f'{self.run}_{self.feature_tag}_appear.csv')
        appear_df = preprocess_appear(appear_csv)
        self.df_dict['appear_post'] = appear_df
        return appear_df

    def preprocess_optical_feature(self) -> pd.DataFrame:
        logger.info('Processing Optical features...')
        optical_csv = os.path.join(self.optical_csv_dir, f'{self.run}_{self.feature_tag}_video_features.csv')
        optical_df = preprocess_optical(optical_csv, standardize=True)
        self.df_dict['optical_post'] = optical_df
        return optical_df

    def pre_process_skel_feature(self) -> pd.DataFrame:
        logger.info('Processing Skel features...')
        skel_csv = os.path.join(self.skel_csv_dir, f'{self.run}_{self.feature_tag}_skel_features.csv')
        skel_df, defective = preprocess_skel(skel_csv, use_position=int(self.use_skel_position),
                                             standardize=True, feature_tag=self.feature_tag,
                                             ratio_samples=self.ratio_samples, ratio_features=self.ratio_features,
                                             stats_skel_csv=self.stats_skel_csv)
        if defective:
            with open(self.filtered_txt, 'a') as f:
                f.write(f"{self.run}\n")
        self.df_dict['skel_post'] = skel_df
        return skel_df

    def preprocess_attention_scene_feature(self) -> pd.DataFrame:
        """
        This method loads the attention weights CSV (for the video given by self.video_name),
        computes for each frame a weighted average GloVe embedding across all objects, and saves
        the result as a CSV. The computed scene vectors are stored in df_dict.
        """
        logger.info("Processing Attention Scene features...")
        attention_csv = os.path.join(self.attention_weights_dir, f"{self.video_name}_attention_weights.csv")
        logger.info(f"Using attention weights from: {attention_csv}")

       
        attn_df = pd.read_csv(attention_csv) #these are the attention weights 
        grouped = attn_df.groupby('frame')
        results = []
        for frame, group in grouped:
            avg_vec = process_frame(group, self.glove_vectors, self.emb_dim)
            if frame % 1000 == 0:
                logger.info(f"Processed frame {frame}")
            results.append((frame, avg_vec))

        results.sort(key=lambda x: x[0])
        out_dict = {'frame': [frame for frame, _ in results]}
        for i in range(self.emb_dim):
            col_name = f'attention_{i+1}'
            out_dict[col_name] = [vec[i] for _, vec in results]
        attention_scene_df = pd.DataFrame(out_dict)

        # output_csv = os.path.join(self.attention_output_dir, f"{self.video_name}_scene_vectors.csv")
        # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        # attention_scene_df.to_csv(output_csv, index=False)
        # logger.info(f"Saved scene vectors to {output_csv}")
        self.df_dict['attention_scene_post'] = attention_scene_df
        return attention_scene_df

    def pre_process_all_features(self) -> None:
        """
        This method loads individual features then combines and temporally aligns them.
        The final combined DataFrame includes appear, optical, skel, and attention-weighted scene features.
        """
        skel_df = self.pre_process_skel_feature()
        appear_df = self.preprocess_appear_feature()
        optical_df = self.preprocess_optical_feature()
        attention_scene_df = self.preprocess_attention_scene_feature()
        combined_resampled_df = self.combine_dataframes([appear_df, optical_df, skel_df, attention_scene_df])
        self.df_dict['combined_resampled_df'] = combined_resampled_df

    def save_df_dict(self) -> None:
        if not os.path.exists(f'{self.out_preprocess_pkl}'):
            os.mkdir(f'{self.out_preprocess_pkl}')
        out_pkl_file = os.path.join(self.out_preprocess_pkl, f"{self.video_name}_{self.feature_tag}.pkl")
        logger.info(f"Saving {out_pkl_file}")
        with open(out_pkl_file, 'wb') as f:
            pkl.dump(self.df_dict, f)
        logger.info(f"Saved {out_pkl_file}")
        with open(self.complete_txt, 'a') as f:
            f.write(f"{self.video_name}\n")



if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config: {args}')
    assert '.txt' not in args.run, f"run argument should be a video name, e.g. 1.2.3_kinect, fed {args.run}"
 
    with open(f'{args.glove}', 'rb') as f:
        glove_vectors = pkl.load(f)

    emb_dim = glove_vectors['apple'].shape[0]
    logger.info(f"GloVe embedding dimension: {emb_dim}")

    processor = FeatureProcessor(configs=args, glove_vectors=glove_vectors, emb_dim=emb_dim)
    try:
        processor.pre_process_all_features()
        processor.save_df_dict()
    except Exception as e:
        with open(processor.error_txt, 'a') as err_f:
            err_f.write(f"{processor.run}\n{traceback.format_exc()}\n")
