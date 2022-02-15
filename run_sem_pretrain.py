seed = 1111
print(f'Setting seeds {seed}')
import random

random.seed(seed)
import numpy as np

np.random.seed(seed)
import tensorflow as tf

tf.random.set_seed(seed)
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import math
import os
import json
import sys
import traceback
import csv
import pickle as pkl
import re
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats

import ray

# this setting seems to limit # active threads for each ray actor method.
# uncomment this line to run cached features
ray.init(num_cpus=12)

sys.path.append('../pysot')
sys.path.append('../SEM2')
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore
from sem.event_models import GRUEvent, LinearEvent, LSTMEvent
from sem.sem import SEM
from utils import SegmentationVideo, get_binned_prediction, get_point_biserial, \
    logger, parse_config, contain_substr, ReadoutDataframes, Sampler, get_coverage, get_purity, event_label_to_interval
from joblib import Parallel, delayed
import gensim.downloader
from typing import List, Dict
from copy import deepcopy

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
with open('gen_sim_glove_50.pkl', 'rb') as f:
    glove_vectors = pkl.load(f)
# glove_vectors = gensim.downloader.load('word2vec-ruscorpora-300')
emb_dim = glove_vectors['apple'].size


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(float)
    return appear_df


def preprocess_optical(vid_csv, standardize=True):
    vid_df = pd.read_csv(vid_csv, index_col='frame')
    # vid_df.drop(['pixel_correlation'], axis=1, inplace=True)
    for c in vid_df.columns:
        if not standardize:
            vid_df.loc[:, c] = (vid_df[c] - min(vid_df[c].dropna())) / (
                    max(vid_df[c].dropna()) - min(vid_df[c].dropna()))
        else:
            vid_df.loc[:, c] = (vid_df[c] - vid_df[c].dropna().mean()) / vid_df[
                c].dropna().std()
    return vid_df


def preprocess_skel(skel_csv, use_position=0, standardize=True):
    skel_df = pd.read_csv(skel_csv, index_col='frame')
    skel_df.drop(['sync_time', 'raw_time', 'body', 'J1_dist_from_J1', 'J1_3D_rel_X', 'J1_3D_rel_Y', 'J1_3D_rel_Z'], axis=1,
                 inplace=True)
    if use_position:
        keeps = ['accel', 'speed', 'dist', 'interhand', '2D', 'rel']
    else:
        keeps = ['accel', 'speed', 'dist', 'interhand', 'rel']

    # for c in skel_df.columns:
    #     if contain_substr(c, keeps):
    #         if not standardize:
    #             skel_df.loc[:, c] = (skel_df[c] - min(skel_df[c].dropna())) / (
    #                     max(skel_df[c].dropna()) - min(skel_df[c].dropna()))
    #         else:
    #             skel_df.loc[:, c] = (skel_df[c] - skel_df[c].dropna().mean()) / skel_df[
    #                 c].dropna().std()
    #     else:
    #         skel_df.drop([c], axis=1, inplace=True)
    #
    # return skel_df

    for c in skel_df.columns:
        if contain_substr(c, keeps):
            continue
        else:
            skel_df.drop([c], axis=1, inplace=True)
    if standardize:
        # load sampled skel features, 200 samples for each video.
        combined_runs = pd.read_csv('sampled_skel_features_dec_6.csv')
        # mask outliers with N/A
        select_indices = (skel_df < combined_runs.quantile(.95)) & (skel_df > combined_runs.quantile(.05))
        skel_df = skel_df[select_indices]
        qualified_columns = (select_indices.sum() > int(len(skel_df) * 0.8))
        assert qualified_columns.sum() / len(qualified_columns) > 0.9, \
            f"Video {skel_csv} has {len(qualified_columns) - qualified_columns.sum()} un-qualified columns!!!"
        # fill N/A
        skel_df = skel_df.ffill()

        # standardize using global statistics
        select_indices = (combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))
        combined_runs_q = combined_runs[select_indices]
        stats = combined_runs_q.describe().loc[['mean', 'std']]
        skel_df = (skel_df - stats.loc['mean', skel_df.columns]) / stats.loc['std', skel_df.columns]

    return skel_df


def remove_number(string):
    for i in range(100):
        string = string.replace(str(i), '')
    return string


def get_emb(category_weights, emb_dim):
    average = np.zeros(shape=(1, emb_dim))
    for category, prob in category_weights.iteritems():
        r = np.zeros(shape=(1, emb_dim))
        try:
            r += glove_vectors[category]
        except Exception as e:
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        average += r * prob
    return average


def get_emb_distance(category_distances, emb_dim=100):
    # Add 1 to avoid 3 objects with 0 distances (rare but might happen), then calculate inverted weights
    category_distances = category_distances + 1
    # Add 1 to avoid cases there is only one object
    category_weights = 1 - category_distances / (category_distances.sum() + 1)
    if category_weights.sum() == 0:
        logger.error('Sum of probabilities is zero')
    average = get_emb(category_weights, emb_dim=emb_dim)
    return average / category_weights.sum()


def get_embs_and_categories(objhand_df: pd.DataFrame, emb_dim=100, num_objects=3):
    obj_handling_embs = np.zeros(shape=(0, emb_dim))
    categories = pd.DataFrame()

    for i, row in objhand_df.iterrows():
        all_categories = list(row.index[row.notna()])
        if len(all_categories):
            # pick the nearest object
            nearest = row.argmin()
            assert nearest != -1
            # obj_handling_emb = get_emb_category(row.index[nearest], emb_dim)
            new_row = pd.Series(data=row.dropna().sort_values().index[:num_objects], name=row.name)
            categories = categories.append(new_row)
            # Interestingly, some words such as towel are more semantically central than mouthwash
            # glove.most_similar(glove['towel'] + ['mouthwash']) yields towel and words close to mouthwash, but not mouthwash!
            obj_handling_emb = get_emb_distance(row.dropna().sort_values()[:num_objects], emb_dim)
        else:
            obj_handling_emb = np.full(shape=(1, emb_dim), fill_value=np.nan)
        obj_handling_embs = np.vstack([obj_handling_embs, obj_handling_emb])
    return obj_handling_embs, categories


def preprocess_objhand(objhand_csv, standardize=True, use_depth=False, num_objects=3, feature='objhand'):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')

    def filter_objhand():
        if use_depth:
            filtered_df = objhand_df.filter(regex=f'_dist_z$')
        else:
            filtered_df = objhand_df.filter(regex=f'_dist$')
        # be careful that filter function return a view, thus filtered_df is a view of objhand_df.
        # deepcopy to avoid unwanted bugs
        filtered_df = deepcopy(filtered_df)
        s = [re.split('([a-zA-Z\s\(\)]+)([0-9]+)', x)[1] for x in filtered_df.columns]
        instances = set(s)
        for i in instances:
            filtered_df.loc[:, i + '_mindist'] = filtered_df[[col for col in filtered_df.columns if i in col]].min(axis=1)
        filtered_df = filtered_df.filter(regex='_mindist')
        # remove mindist
        filtered_df.rename(lambda x: x.replace('_mindist', ''), axis=1, inplace=True)

        return filtered_df

    objhand_df = filter_objhand()

    obj_handling_embs, categories = get_embs_and_categories(objhand_df, emb_dim=emb_dim, num_objects=num_objects)

    obj_handling_embs = pd.DataFrame(obj_handling_embs, index=objhand_df.index,
                                     columns=list(map(lambda x: f'{feature}_{x}', range(emb_dim))))
    # Standardizing using a single video might project embedded vectors to weird space (e.g. mouthwash -> srishti)
    # Moreover, the word2vec model already standardize for the whole corpus, thus we don't need to standardize.
    if standardize:
        obj_handling_embs = (obj_handling_embs - obj_handling_embs.mean()) / obj_handling_embs.std()

    return obj_handling_embs, categories


def get_emb_speed(categories_speed, emb_dim):
    if categories_speed.sum() == 0:
        logger.error('Sum of probabilities is zero')
    average = get_emb(categories_speed, emb_dim)
    # Not dividing categories_speed here (scene-based) because we care about relative values between scenes.
    return average


def get_embs(objspeed_df: pd.DataFrame, emb_dim=100):
    objspeed_embeddings = np.zeros(shape=(0, emb_dim))
    for index, row in objspeed_df.iterrows():
        objspeed_emb = get_emb_speed(row.dropna(), emb_dim)
        objspeed_embeddings = np.vstack([objspeed_embeddings, objspeed_emb])
    return objspeed_embeddings


def preprocess_objspeed(objspeed_csv, standardize=True):
    objspeed_df = pd.read_csv(objspeed_csv, index_col='frame')
    # remove _maxspeed
    objspeed_df.rename(lambda x: x.replace('_maxspeed', ''), axis=1, inplace=True)
    objspeed_df.dropna(axis=0, how='all', inplace=True)
    objspeed_embeddings = get_embs(objspeed_df, emb_dim)

    objspeed_embeddings = pd.DataFrame(objspeed_embeddings, index=objspeed_df.index,
                                       columns=list(map(lambda x: f'objspeed_{x}', range(emb_dim))))
    if standardize:
        objspeed_embeddings = (objspeed_embeddings - objspeed_embeddings.mean()) / objspeed_embeddings.std()

    return objspeed_embeddings


def interpolate_frame(dataframe: pd.DataFrame):
    first_frame = dataframe.index[0]
    last_frame = dataframe.index[-1]
    dummy_frame = pd.DataFrame(np.NaN, index=range(first_frame, last_frame),
                               columns=dataframe.columns)
    dummy_frame = dummy_frame.combine_first(dataframe).interpolate(limit_area='inside')
    return dummy_frame


def pca_dataframe(dataframe: pd.DataFrame):
    dataframe.dropna(axis=0, inplace=True)
    pca = PCA(args.pca_explained, whiten=True)
    dummy_array = pca.fit_transform(dataframe.values)

    return pd.DataFrame(dummy_array)


def combine_dataframes(data_frames, rate='40ms', fps=30):
    # Some features such as optical flow are calculated not for all frames, interpolate first
    data_frames = [interpolate_frame(df) for df in data_frames]
    combine_df = pd.concat(data_frames, axis=1)
    # After dropping null values, variances are not unit anymore, some are around 0.8.
    combine_df.dropna(axis=0, inplace=True)
    first_frame = combine_df.index[0]
    combine_df['frame'] = combine_df.index
    # After resampling, some variances drop to 0.3 or 0.4
    combine_df = resample_df(combine_df, rate=rate, fps=fps)
    # because resample use mean, need to adjust categorical variables
    combine_df['appear'] = combine_df['appear'].apply(math.ceil).astype(float)
    combine_df['disappear'] = combine_df['disappear'].apply(math.ceil).astype(float)
    # Add readout to visualize
    data_frames = [combine_df[df.columns] for df in data_frames]
    for df in data_frames:
        df.index = combine_df['frame'].apply(round)

    assert combine_df.isna().sum().sum() == 0
    combine_df.drop(['sync_time', 'frame'], axis=1, inplace=True, errors='ignore')
    return combine_df, first_frame, data_frames


def plot_subject_model_boundaries(gt_freqs, pred_boundaries, title='', save_fig=True,
                                  show=True, bicorr=0.0, percentile=0.0):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.title(title)
    b = np.arange(len(pred_boundaries))[pred_boundaries][0]
    plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75, color='b')
    for b in np.arange(len(pred_boundaries))[pred_boundaries][1:]:
        plt.plot([b, b], [0, 1], 'k:', alpha=0.75, color='b')

    plt.text(0.1, 0.3, f'bicorr={bicorr:.3f}, perc={percentile:.3f}', fontsize=14)
    plt.legend(loc='upper left')
    plt.ylim([0, 0.4])
    sns.despine()
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()
    plt.close()


def plot_diagnostic_readouts(gt_freqs, sem_readouts, frame_interval=3.0, offset=0.0, title='', show=False, save_fig=True,
                             bicorr=0.0, percentile=0.0, pearson_r=0.0):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.ylim([0, 0.4])
    plt.title(title)
    plt.text(0.1, 0.3, f'bicorr={bicorr:.3f}, perc={percentile:.3f}, pearson={pearson_r:.3f}', fontsize=14)
    colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}

    cm = plt.get_cmap('gist_rainbow')
    post = sem_readouts.e_hat
    boundaries = sem_readouts.boundaries
    NUM_COLORS = post.max()
    # Hard-code 40 events for rainbow to be able to compare across events
    # NUM_COLORS = 30
    """
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
    """
    for i, (b, e) in enumerate(zip(boundaries, post)):
        if b != 0:
            second = i / frame_interval + offset
            if b == 1:
                plt.axvline(second, linestyle=(0, (5, 10)), alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Old Event')
            elif b == 2:
                plt.axvline(second, linestyle='solid', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='New Event')
            elif b == 3:
                plt.axvline(second, linestyle='dotted', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Restart Event')
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(vmin=0, vmax=NUM_COLORS, clip=False)),
                 orientation='horizontal')
    from matplotlib.lines import Line2D
    linestyles = ['dashed', 'solid', 'dotted']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=ls) for ls in linestyles]
    labels = ['Old Event', 'New Event', 'Restart Event']
    plt.legend(lines, labels)
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()
    plt.close()


def plot_pe(sem_readouts, frame_interval, offset, title):
    fig, ax = plt.subplots()
    df = pd.DataFrame({'prediction_error': sem_readouts.pe}, index=range(len(sem_readouts.pe)))
    df['second'] = df.index / frame_interval + offset
    df.plot(kind='line', x='second', y='prediction_error', alpha=1.00, ax=ax)
    plt.savefig('output/run_sem/' + title + '.png')
    plt.close()


def resample_df(objhand_df, rate='40ms', fps=30):
    # fps matter hear, we need feature vector at anchor timepoints to correspond to segments
    outdf = objhand_df.set_index(pd.to_datetime(objhand_df.index / fps, unit='s'), drop=False,
                                 verify_integrity=True)
    resample_index = pd.date_range(start=outdf.index[0], end=outdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=outdf.columns)
    outdf = outdf.combine_first(dummy_frame).interpolate(method='time', limit_area='inside').resample(rate).mean()
    return outdf


def merge_feature_lists(txt_out):
    with open('appear_complete.txt', 'r') as f:
        appears = f.readlines()

    with open('vid_complete.txt', 'r') as f:
        vids = f.readlines()

    with open('skel_complete.txt', 'r') as f:
        skels = f.readlines()

    with open('objhand_complete.txt', 'r') as f:
        objhands = f.readlines()

    sem_runs = set(appears).intersection(set(skels)).intersection(set(vids)).intersection(
        set(objhands))
    with open(txt_out, 'w') as f:
        f.writelines(sem_runs)


class SEMContext:
    """
    This class maintain global variables for SEM training and inference
    """

    def __init__(self, sem_model=None, run_kwargs=None, tag='', configs=None, sampler=None):
        self.sem_model = sem_model
        self.run_kwargs = run_kwargs
        self.tag = tag
        if not os.path.exists(f'output/run_sem/{self.tag}'):
            os.makedirs(f'output/run_sem/{self.tag}')
        self.configs = configs
        self.epochs = int(self.configs.epochs)
        self.train_stratified = int(self.configs.train_stratified)
        self.second_interval = 1
        self.sample_per_second = 1000 / float(self.configs.rate.replace('ms', ''))

        self.current_epoch = None
        # These variables will be set based on each run
        self.run = None
        self.movie = None
        self.title = None
        self.grain = None
        # FPS is used to pad prediction boundaries, should be inferred from run
        self.fps = None
        self.is_train = None  # this variable is set depending on mode
        # These variables will be set by read_train_valid_list
        self.train_list = None
        self.train_dataset = None
        self.valid_dataset = None
        # These variables will bet set after processing features
        self.first_frame = None
        self.last_frame = None
        self.end_second = None
        self.data_frames = None
        self.combine_df = None
        self.categories = None
        self.categories_z = None

        self.sampler = sampler
        self.chapters = [4, 2, 1, 3]
        # if self.sampler is not None:
        #     logger.info(f'Number of Epochs from Sampler={self.sampler.max_epoch}, not from Config={self.epochs}')
        #     self.epochs = self.sampler.max_epoch

    def iterate(self, is_eval=True):
        for e in range(self.epochs):
            # epoch counting from 1 for inputdf and diagnostic.
            self.current_epoch = e + 1
            logger.info('Training')
            if self.train_stratified:
                self.train_dataset = self.train_list[e % 8]
            else:
                self.train_dataset = self.train_list
            self.training()
            if is_eval and self.current_epoch % 5 == 1:
            # if is_eval and self.current_epoch % 5 == 0:
                logger.info('Evaluating')
                self.evaluating()
            # break
            # LR annealing
            # if self.current_epoch % 10 == 0:
            #     self.sem_model.general_event_model.decrease_lr()
            #     self.sem_model.general_event_model_x2.decrease_lr()
            #     self.sem_model.general_event_model_x3.decrease_lr()
            #     self.sem_model.general_event_model_yoke.decrease_lr()
            #     for k, e in self.sem_model.event_models.items():
            #         e.decrease_lr.remote()

    def training(self):
        # TODO: can be refactored for simplicity
        if self.sampler is not None:
            logger.info('Using sampler instead of train.txt!!!')
            # for c in self.chapters:
            run = self.sampler.get_one_run()
            self.is_train = True
            self.sem_model.kappa = float(self.configs.kappa)
            self.sem_model.alfa = float(self.configs.alfa)
            logger.info(f'Training video {run}')
            self.set_run_variables(run)
            self.infer_on_video(store_dataframes=int(self.configs.store_frames))
            # The order of percentile seems to be 4231.
            # self.chapters = np.random.permutation(self.chapters)

        else:
            # Randomize order of video
            random.shuffle(self.train_dataset)
            # self.train_dataset = np.random.permutation(self.train_dataset)
            run = self.train_dataset[(self.current_epoch - 1) % len(self.train_dataset)]
            self.is_train = True
            self.sem_model.kappa = float(self.configs.kappa)
            self.sem_model.alfa = float(self.configs.alfa)
            logger.info(f'Training video {run} at epoch {self.current_epoch}')
            self.set_run_variables(run)
            self.infer_on_video(store_dataframes=int(self.configs.store_frames))

    def evaluating(self):
        # Randomize order of video
        # random.shuffle(self.valid_dataset)
        self.valid_dataset = np.random.permutation(self.valid_dataset)
        for index, run in enumerate(self.valid_dataset):
            self.is_train = False
            self.sem_model.kappa = 0
            self.sem_model.alfa = 1e-30
            logger.info(f'Evaluating video {run} at epoch {self.current_epoch}')
            self.set_run_variables(run)
            self.infer_on_video(store_dataframes=int(self.configs.store_frames))
            # break

    def parse_input(self, token, is_stratified=0) -> List:
        # Whether the train.txt or 4.4.4_kinect
        if '.txt' in token:
            with open(token, 'r') as f:
                list_input = f.readlines()
                list_input = [x.strip() for x in list_input]
            if is_stratified:
                assert '.txt' in list_input[0], f"Attempting to stratify individual {list_input}"
                return [self.parse_input(i) for i in list_input]
            else:
                # sum(list, []) to remove nested list
                return sum([self.parse_input(i) for i in list_input], [])
        else:
            # To be consistent when config is stratified
            if is_stratified:
                return [[token]]
            # degenerate case, e.g. 4.4.4_kinect
            return [token]

    def read_train_valid_list(self):
        self.valid_dataset = self.parse_input(self.configs.valid)
        self.train_list = self.parse_input(self.configs.train, is_stratified=self.train_stratified)

    def set_run_variables(self, run):
        self.run = run
        self.movie = run + '_trim.mp4'
        self.title = os.path.join(self.tag, os.path.basename(self.movie[:-4]) + self.tag)
        # self.grain = 'coarse'
        # FPS is used to pad prediction boundaries, should be inferred from run
        if 'kinect' in run:
            self.fps = 25
        else:
            self.fps = 30

    def infer_on_video(self, store_dataframes=1):
        try:
            self.process_features(use_cache=int(self.configs.use_cache), cache_tag=self.configs.cache_tag)

            if store_dataframes:
                # Infer coordinates from nearest categories and add both to data_frame for visualization
                # objhand_csv = os.path.join(self.configs.objhand_csv, self.run + '_objhand.csv')
                # objhand_df = pd.read_csv(objhand_csv)
                # objhand_df = objhand_df.loc[self.data_frames.skel_post.index, :]

                def add_category_and_coordinates(categories: pd.DataFrame, use_depth=False):
                    # Readout to visualize object-hand features
                    # Re-index: some frames there are no objects near hand (which is not possible, this bug is due to min(89, NaN)=NaN
                    # categories = categories.reindex(range(categories.index[-1])).ffill()
                    categories = categories.ffill()
                    categories = categories.loc[self.data_frames.skel_post.index, :]
                    setattr(self.data_frames, 'categories' + ('_z' if use_depth else ''), categories)
                    # coordinates variable is determined by categories variable, thus having only 3 objects
                    coordinates = pd.DataFrame()
                    # No need to construct coordinates, save time processing.
                    # for index, r in categories.iterrows():
                    #     frame_series = pd.Series(dtype=float)
                    #     # There could be a case where there are two spray bottles near the hand: 6.3.6
                    #     # When num_objects is large, there are nan in categories -> filter
                    #     all_categories = set(r.dropna().values)
                    #     for c in list(all_categories):
                    #         # Filter by category name and select distance
                    #         # Note: paper towel and towel causes duplicated columns in series,
                    #         # Need anchor ^ to distinguish towel and paper towel (2.4.7),
                    #         # need digit \d to distinguish pillow0 and pillowcase0 (3.3.5)
                    #         # Need to escape character ( and ) in aloe (green bottle) (4.4.5)
                    #         # Either use xy or depth (z) distance to get nearest object names
                    #         if use_depth:
                    #             df = objhand_df.loc[index, :].filter(regex=f"^{re.escape(c)}\d").filter(regex='_dist_z$')
                    #             nearest_name = df.index[df.argmin()].replace('_dist_z',
                    #                                                          '')  # e.g. pillow0, pillowcase0, towel0, paper towel0
                    #         else:
                    #             df = objhand_df.loc[index, :].filter(regex=f"^{re.escape(c)}\d").filter(regex='_dist$')
                    #             nearest_name = df.index[df.argmin()].replace('_dist',
                    #                                                          '')  # e.g. pillow0, pillowcase0, towel0, paper towel0
                    #         # select nearest object's coordinates
                    #         # need anchor ^ to distinguish between towel0 and paper towel0
                    #         s = objhand_df.loc[index, :].filter(regex=f"^{re.escape(nearest_name)}")
                    #         frame_series = frame_series.append(s)
                    #     frame_series.name = index
                    #     coordinates = coordinates.append(frame_series)
                    setattr(self.data_frames, 'coordinates' + ('_z' if use_depth else ''), coordinates)

                # add_category_and_coordinates(self.categories, use_depth=False)
                # Adding categories_z and coordinates_z to data_frame
                add_category_and_coordinates(self.categories_z, use_depth=True)

            # logger.info(f'Features: {self.combine_df.columns}')
            # Note that without copy=True, this code will return a view and subsequent changes to x_train will change self.combine_df
            # e.g. x_train /= np.sqrt(x_train.shape[1]) or x_train[2] = ...
            x_train = self.combine_df.to_numpy(copy=True)
            # PCA transform input features. Also, get inverted vector for visualization
            if int(self.configs.pca):
                if int(self.configs.use_ind_feature_pca):
                    # TODO: cross check with get_pca_from_all_runs.py to make sure features' positions are correct
                    pca_appear = pkl.load(open(f'{self.configs.pca_tag}_appear_pca.pkl', 'rb'))
                    x_train_pca_appear = pca_appear.transform(x_train[:, :2])
                    pca_optical = pkl.load(open(f'{self.configs.pca_tag}_optical_pca.pkl', 'rb'))
                    x_train_pca_optical = pca_optical.transform(x_train[:, 2:4])
                    pca_skel = pkl.load(open(f'{self.configs.pca_tag}_skel_pca.pkl', 'rb'))
                    x_train_pca_skel = pca_skel.transform(x_train[:, 4:-100])
                    pca_emb = pkl.load(open(f'{self.configs.pca_tag}_emb_pca.pkl', 'rb'))
                    x_train_pca_emb = pca_emb.transform(x_train[:, -100:])
                    x_train_pca = np.hstack([x_train_pca_appear, x_train_pca_optical, x_train_pca_skel, x_train_pca_emb])

                    indices = [pca_appear.n_components,
                               pca_appear.n_components + pca_optical.n_components,
                               pca_appear.n_components + pca_optical.n_components + pca_skel.n_components,
                               pca_appear.n_components + pca_optical.n_components + pca_skel.n_components + pca_emb.n_components]
                    x_train_inverted_appear = pca_appear.inverse_transform(x_train_pca[:, :indices[0]])
                    x_train_inverted_optical = pca_optical.inverse_transform(x_train_pca[:, indices[0]:indices[1]])
                    x_train_inverted_skel = pca_skel.inverse_transform(x_train_pca[:, indices[1]:indices[2]])
                    x_train_inverted_emb = pca_emb.inverse_transform(x_train_pca[:, indices[2]:])
                    x_train_inverted = np.hstack(
                        [x_train_inverted_appear, x_train_inverted_optical, x_train_inverted_skel, x_train_inverted_emb])
                else:
                    pca = pkl.load(open(f'{self.configs.pca_tag}_pca.pkl', 'rb'))
                    if x_train.shape[1] != pca.n_features_:
                        logger.error(
                            f'MISMATCH: pca.n_features_ = {pca.n_features_} vs. input features={x_train.shape[1]}!!!')
                        raise
                    x_train_pca = pca.transform(x_train)

                    x_train_inverted = pca.inverse_transform(x_train_pca)
                x_train = x_train_pca
                df_x_train = pd.DataFrame(data=x_train, index=self.data_frames.skel_post.index)
                setattr(self.data_frames, 'x_train_pca', df_x_train)
                df_x_train_inverted = pd.DataFrame(data=x_train_inverted, index=self.data_frames.skel_post.index,
                                                   columns=self.combine_df.columns)
                setattr(self.data_frames, 'x_train_inverted', df_x_train_inverted)
            else:
                df_x_train = pd.DataFrame(data=x_train, index=self.data_frames.skel_post.index, columns=self.combine_df.columns)
                setattr(self.data_frames, 'x_train', df_x_train)

            # Note that this is different from x_train = x_train / np.sqrt(x_train.shape[1]), the below code will change values of
            # the memory allocation, to which self.combine_df refer -> change to be safer
            # x_train /= np.sqrt(x_train.shape[1])
            # x_train is already has unit variance for all features (pca whitening) -> scale to have unit length.
            # In SEM's comment, it should be useful to have unit length stimulus.
            x_train = x_train / np.sqrt(x_train.shape[1])
            # x_train = np.random.permutation(x_train)
            # Comment this chunk to generate cached features faster
            # This function train and change sem event models
            self.run_sem_and_plot(x_train)
            if store_dataframes:
                # Transform predicted vectors to the original vector space for visualization
                if int(self.configs.pca):
                    x_inferred_pca = self.sem_model.results.x_hat
                    # Scale back to PCA whitening results
                    x_inferred_pca = x_inferred_pca * np.sqrt(x_train.shape[1])
                    df_x_inferred = pd.DataFrame(data=x_inferred_pca, index=self.data_frames.skel_post.index)
                    setattr(self.data_frames, 'x_inferred_pca', df_x_inferred)
                    if int(self.configs.use_ind_feature_pca):
                        indices = [pca_appear.n_components,
                                   pca_appear.n_components + pca_optical.n_components,
                                   pca_appear.n_components + pca_optical.n_components + pca_skel.n_components,
                                   pca_appear.n_components + pca_optical.n_components + pca_skel.n_components + pca_emb.n_components]
                        x_inferred_inverted_appear: np.ndarray = pca_appear.inverse_transform(x_inferred_pca[:, :indices[0]])
                        x_inferred_inverted_optical = pca_optical.inverse_transform(x_inferred_pca[:, indices[0]:indices[1]])
                        x_inferred_inverted_skel = pca_skel.inverse_transform(x_inferred_pca[:, indices[1]:indices[2]])
                        x_inferred_inverted_emb = pca_emb.inverse_transform(x_inferred_pca[:, indices[2]:])
                        x_inferred_inverted = np.hstack(
                            [x_inferred_inverted_appear, x_inferred_inverted_optical, x_inferred_inverted_skel,
                             x_inferred_inverted_emb])
                    else:
                        x_inferred_inverted = pca.inverse_transform(x_inferred_pca)
                    df_x_inferred_inverted = pd.DataFrame(data=x_inferred_inverted, index=self.data_frames.skel_post.index,
                                                          columns=self.combine_df.columns)
                    setattr(self.data_frames, 'x_inferred_inverted', df_x_inferred_inverted)
                else:
                    x_inferred_ori = self.sem_model.results.x_hat * np.sqrt(x_train.shape[1])
                    df_x_inferred_ori = pd.DataFrame(data=x_inferred_ori, index=self.data_frames.skel_post.index,
                                                     columns=self.combine_df.columns)
                    setattr(self.data_frames, 'x_inferred', df_x_inferred_ori)

                with open('output/run_sem/' + self.title + f'_inputdf_{self.current_epoch}.pkl', 'wb') as f:
                    pkl.dump(self.data_frames, f)
            # Uncomment this chunk and comment the above chunk to save cached features faster, also no eval
            # with open('output/run_sem/' + self.title + f'_inputdf_{self.current_epoch}.pkl', 'wb') as f:
            #     pkl.dump(self.data_frames, f)

            logger.info(f'Done SEM {self.run}. is_train={self.is_train}!!!\n')
            with open('output/run_sem/sem_complete.txt', 'a') as f:
                f.write(self.run + f'_{tag}' + '\n')
            # sem's Results() is initialized and different for each run
        except Exception as e:
            with open('output/run_sem/sem_error.txt', 'a') as f:
                f.write(self.run + f'_{tag}' + '\n')
                f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc())

    def process_features(self, use_cache=0, cache_tag=''):
        """
        This method load pre-processed features then combine and align them temporally
        :return:
        """
        if use_cache:
            readout_dataframes = pkl.load(open(f'output/run_sem/{cache_tag}/{self.movie[:-4]}{cache_tag}_inputdf_1.pkl', 'rb'))

            appear_df = readout_dataframes.appear_post
            optical_df = readout_dataframes.optical_post
            skel_df = readout_dataframes.skel_post
            obj_handling_embs = readout_dataframes.objhand_post
            scene_embs = readout_dataframes.scene_post
            # categories = readout_dataframes.categories
            categories_z = readout_dataframes.categories_z

            # switch to scene motion
            # objspeed_embs = readout_dataframes.objspeed_post
            # data_frames = [appear_df, optical_df, skel_df, obj_handling_embs,
            #                objspeed_embs]
            # TODO: switch to 5 components with two features
            # data_frames = [skel_df, obj_handling_embs]
            data_frames = [appear_df, optical_df, skel_df, obj_handling_embs, scene_embs]
            combine_df = pd.concat(data_frames, axis=1)
            first_frame = appear_df.index[0]

        else:
            # For some reason, some optical flow videos have inf value
            pd.set_option('use_inf_as_na', True)

            logger.info(f'Loading features from csv formats')
            objhand_csv = os.path.join(self.configs.objhand_csv, self.run + '_objhand.csv')
            skel_csv = os.path.join(self.configs.skel_csv, self.run + '_skel_features.csv')
            appear_csv = os.path.join(self.configs.appear_csv, self.run + '_appear.csv')
            optical_csv = os.path.join(self.configs.optical_csv, self.run + '_video_features.csv')
            # objspeed_csv = os.path.join(self.configs.objspeed_csv, self.run + '_objspeed.csv')

            logger.info(f'Processing features...')
            # Load csv files and preprocess to get a scene vector
            logger.info(f'Processing Appear features...')
            appear_df = preprocess_appear(appear_csv)
            logger.info(f'Processing Skel features...')
            skel_df = preprocess_skel(skel_csv, use_position=int(self.configs.use_position), standardize=True)
            logger.info(f'Processing Optical features...')
            optical_df = preprocess_optical(optical_csv, standardize=True)
            logger.info(f'Processing Objhand features...')
            # Switch obj_handling_embs appropriately to use depth
            # _, categories = preprocess_objhand(objhand_csv, standardize=True,
            #                                    num_objects=int(self.configs.num_objects),
            #                                    use_depth=False)
            obj_handling_embs, categories_z = preprocess_objhand(objhand_csv, standardize=False,
                                                                 num_objects=int(self.configs.num_objects),
                                                                 use_depth=True, feature='objhand')
            logger.info(f'Processing Scene features...')
            scene_embs, _ = preprocess_objhand(objhand_csv, standardize=False, num_objects=30, use_depth=True,
                                               feature='scene')
            # logger.info(f'Processing Objspeed features...')
            # objspeed_embs = preprocess_objspeed(objspeed_csv, standardize=True)

            # Get consistent start-end times and resampling rate for all features
            # TODO: switch to 5 components with two features
            # combine_df, first_frame, data_frames = combine_dataframes([skel_df, obj_handling_embs],
            #                                                           rate=self.configs.rate, fps=self.fps)
            readout_dataframes = ReadoutDataframes()
            setattr(readout_dataframes, 'objhand_pre', obj_handling_embs)
            # combine_df, first_frame, data_frames = combine_dataframes([appear_df, optical_df, skel_df, obj_handling_embs],
            #                                                           rate=self.configs.rate, fps=self.fps)
            # Switch to use scene embedding or not
            combine_df, first_frame, data_frames = combine_dataframes([appear_df, optical_df, skel_df, obj_handling_embs,
                                                                       scene_embs],
                                                                      rate=self.configs.rate, fps=self.fps)
        for feature, df in zip(['appear_post', 'optical_post', 'skel_post', 'objhand_post', 'scene_post'], data_frames):
            # for feature, df in zip(['appear_post', 'optical_post', 'skel_post', 'objhand_post'], data_frames):
            setattr(readout_dataframes, feature, df)
        self.last_frame = readout_dataframes.skel_post.index[-1]
        self.first_frame = first_frame
        # This parameter is used to limit the time of ground truth video according to feature data
        self.end_second = math.ceil(self.last_frame / self.fps)
        self.data_frames = readout_dataframes
        self.combine_df = combine_df
        # self.categories = categories
        self.categories_z = categories_z

    def calculate_correlation(self, pred_boundaries, grain='coarse'):
        # Process segmentation data (ground_truth)
        data_frame = pd.read_csv(self.configs.seg_path)
        seg_video = SegmentationVideo(data_frame=data_frame, video_path=self.movie)
        seg_video.get_human_segments(n_annotators=100, condition=grain, second_interval=self.second_interval)
        # Compare SEM boundaries versus participant boundaries
        last = min(len(pred_boundaries), self.end_second)
        # this function aggregate subject boundaries, apply a gaussian kernel and calculate correlations for subjects
        biserials = seg_video.get_biserial_subjects(second_interval=self.second_interval, end_second=last)
        # compute biserial correlation and pearson_r of model boundaries
        bicorr = get_point_biserial(pred_boundaries[:last], seg_video.gt_freqs[:last])
        pred_boundaries_gaussed = gaussian_filter1d(pred_boundaries.astype(float), 2)
        pearson_r, p = stats.pearsonr(pred_boundaries_gaussed[:last], seg_video.gt_freqs[:last])
        percentile = percentileofscore(biserials, bicorr)
        logger.info(f'Tag={tag}: Bicorr={bicorr:.3f} cor. Percentile={percentile:.3f},  '
                    f'Subjects_median={np.nanmedian(biserials):.3f}')

        return bicorr, percentile, pearson_r, seg_video

    def compute_clustering_metrics(self):
        start_second = self.first_frame / self.fps
        event_to_intervals = event_label_to_interval(self.sem_model.results.e_hat, start_second)
        df = pd.read_csv('./event_annotation_timing.csv')
        run_df = df[df['run'] == self.run.split('_')[0]]
        # calculate coverage
        coverage_df = pd.DataFrame(
            columns=['annotated_event', 'annotated_length', 'sem_max_overlap', 'max_coverage', 'epoch', 'run', 'tag', 'is_train'])
        for i, annotations in run_df.iterrows():
            ann_event, max_coverage_event, max_coverage = get_coverage(annotations, event_to_intervals)
            coverage_df.loc[len(coverage_df.index)] = [ann_event, annotations['endsec'] - annotations['startsec'],
                                                       max_coverage_event, max_coverage,
                                                       self.current_epoch, self.run, self.tag, self.is_train]
        # calculate purity
        purity_df = pd.DataFrame(
            columns=['sem_event', 'sem_length', 'annotated_max_overlap', 'max_purity', 'epoch', 'run', 'tag', 'is_train'])
        for sem_event, sem_intervals in event_to_intervals.items():
            sem_event, max_purity_ann_event, max_purity = get_purity(sem_event, sem_intervals, run_df)
            sem_length = sum([interval[1] - interval[0] for interval in sem_intervals])
            purity_df.loc[len(purity_df.index)] = [sem_event, sem_length, max_purity_ann_event, max_purity,
                                                   self.current_epoch, self.run, self.tag, self.is_train]
        purity_df.to_csv(path_or_buf='output/run_sem/purity.csv', index=False, header=False, mode='a')
        coverage_df.to_csv(path_or_buf='output/run_sem/coverage.csv', index=False, header=False, mode='a')
        average_coverage = np.average(coverage_df['max_coverage'], weights=coverage_df['annotated_length'])
        average_purity = np.average(purity_df['max_purity'], weights=purity_df['sem_length'])

        return average_purity, average_coverage

    def run_sem_and_plot(self, x_train):
        """
        This method run SEM and plot
        :param x_train:
        :return:
        """
        self.sem_model.run(x_train, train=self.is_train, **self.run_kwargs)
        # set k_prev to None in order to run the next video
        self.sem_model.k_prev = None
        # set x_prev to None in order to train the general event
        self.sem_model.x_prev = None

        # Process results returned by SEM
        average_purity, average_coverage = self.compute_clustering_metrics()

        # Logging some information about types of boundaries
        switch_old = (self.sem_model.results.boundaries == 1).sum()
        switch_new = (self.sem_model.results.boundaries == 2).sum()
        switch_current = (self.sem_model.results.boundaries == 3).sum()
        logger.info(f'Total # of OLD switches: {switch_old}')
        logger.info(f'Total # of NEW switches: {switch_new}')
        logger.info(f'Total # of RESTART switches: {switch_current}')
        entropy = stats.entropy(self.sem_model.results.c) / np.log((self.sem_model.results.c > 0).sum())

        pred_boundaries = get_binned_prediction(self.sem_model.results.boundaries, second_interval=self.second_interval,
                                                sample_per_second=self.sample_per_second)
        # switching between video, not a real boundary
        pred_boundaries[0] = 0
        # Padding prediction boundaries, could be changed to have higher resolution but not necessary
        pred_boundaries = np.hstack([[0] * round(self.first_frame / self.fps / self.second_interval), pred_boundaries]).astype(
            int)
        logger.info(f'Total # of pred_boundaries: {sum(pred_boundaries)}')
        logger.info(f'Total # of event models: {len(self.sem_model.event_models) - 1}')
        threshold = 600
        active_event_models = np.count_nonzero(self.sem_model.c > threshold)
        logger.info(f'Total # of event models active more than {threshold // 3}s: {active_event_models}')

        mean_pe = self.sem_model.results.pe.mean()
        std_pe = self.sem_model.results.pe.std()
        with open('output/run_sem/results_purity_coverage.csv', 'a') as f:
            writer = csv.writer(f)
            # len adds 1, and the buffer model adds 1 => len() - 2
            bicorr, percentile, pearson_r, seg_video = self.calculate_correlation(pred_boundaries=pred_boundaries, grain='coarse')
            with open('output/run_sem/' + self.title + '_gtfreqs_coarse.pkl', 'wb') as f:
                pkl.dump(seg_video.gt_freqs, f)
            writer.writerow([self.run, 'coarse', bicorr, percentile, len(self.sem_model.event_models) - 2, active_event_models,
                             self.current_epoch, (self.sem_model.results.boundaries != 0).sum(), sem_init_kwargs, tag, mean_pe,
                             std_pe, pearson_r, self.is_train,
                             switch_old, switch_new, switch_current, entropy,
                             average_purity, average_coverage])
            bicorr, percentile, pearson_r, seg_video = self.calculate_correlation(pred_boundaries=pred_boundaries, grain='fine')
            with open('output/run_sem/' + self.title + '_gtfreqs_fine.pkl', 'wb') as f:
                pkl.dump(seg_video.gt_freqs, f)
            writer.writerow([self.run, 'fine', bicorr, percentile, len(self.sem_model.event_models) - 2, active_event_models,
                             self.current_epoch, (self.sem_model.results.boundaries != 0).sum(), sem_init_kwargs, tag, mean_pe,
                             std_pe, pearson_r, self.is_train,
                             switch_old, switch_new, switch_current, entropy,
                             average_purity, average_coverage])

        plot_diagnostic_readouts(seg_video.gt_freqs, self.sem_model.results,
                                 frame_interval=self.second_interval * self.sample_per_second,
                                 offset=self.first_frame / self.fps / self.second_interval,
                                 title=self.title + f'_diagnostic_fine_{self.current_epoch}',
                                 bicorr=bicorr, percentile=percentile, pearson_r=pearson_r)

        plot_pe(self.sem_model.results, frame_interval=self.second_interval * self.sample_per_second,
                offset=self.first_frame / self.fps / self.second_interval,
                title=self.title + f'_PE_fine_{self.current_epoch}')

        # logging results
        with open('output/run_sem/' + self.title + f'_diagnostic_{self.current_epoch}.pkl', 'wb') as f:
            pkl.dump(self.sem_model.results.__dict__, f)
        with open('output/run_sem/' + self.title + '_gtfreqs.pkl', 'wb') as f:
            pkl.dump(seg_video.gt_freqs, f)


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config: {args}')

    if not os.path.exists('output/run_sem/results_purity_coverage.csv'):
        csv_headers = ['run', 'grain', 'bicorr', 'percentile', 'n_event_models', 'active_event_models', 'epoch',
                       'number_boundaries', 'sem_params', 'tag', 'mean_pe', 'std_pe', 'pearson_r', 'is_train',
                       'switch_old', 'switch_new', 'switch_current', 'entropy',
                       'purity', 'coverage']
        with open('output/run_sem/results_purity_coverage.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    if not os.path.exists('output/run_sem/purity.csv'):
        csv_headers = ['sem_event', 'sem_length', 'annotated_max_overlap', 'max_purity', 'epoch', 'run', 'tag', 'is_train']
        with open('output/run_sem/purity.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    if not os.path.exists('output/run_sem/coverage.csv'):
        csv_headers = ['annotated_event', 'annotated_length', 'sem_max_overlap', 'max_coverage', 'epoch', 'run', 'tag',
                       'is_train']
        with open('output/run_sem/coverage.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # Initialize keras model and running
    # Define model architecture, hyper parameters and optimizer
    f_class = GRUEvent
    # f_class = LSTMEvent
    optimizer_kwargs = dict(lr=float(args.lr), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    # these are the parameters for the event model itself.
    f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, dropout=0.5,
                  n_epochs=1, t=4, batch_update=True, n_hidden=int(args.n_hidden), variance_window=None,
                  optimizer_kwargs=optimizer_kwargs)
    # set the hyper parameters for segmentation
    lmda = float(args.lmda)  # stickyness parameter (prior)
    alfa = float(args.alfa)  # concentration parameter (prior)
    kappa = float(args.kappa)
    sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'kappa': kappa, 'f_opts': f_opts,
                       'f_class': f_class}
    logger.info(f'SEM parameters: {sem_init_kwargs}')
    # set default hyper parameters for each run, can be overridden later
    run_kwargs = dict()
    sem_model = SEM(**sem_init_kwargs)
    tag = args.tag

    if int(args.use_sampler):

        df_select = pd.read_csv('output/run_sem/results_corpus.csv')
        df_select['chapter'] = df_select['run'].apply(lambda x: int(x[2]))
        # metrics are read from cache_tag
        interested_tags = [args.cache_tag]
        df_select = df_select[(df_select['tag'].isin(interested_tags)) & (df_select['is_train'] == True)]
        with open(args.valid, 'r') as f:
            valid_runs = f.readlines()
            valid_runs = [x.strip() for x in valid_runs]
        sampler = Sampler(df_select=df_select, validation_runs=valid_runs)
        sampler.prepare_list(min_boundary=5, max_boundary=50, metric='percentile')
    else:
        sampler = None
    context_sem = SEMContext(sem_model=sem_model, run_kwargs=run_kwargs, tag=tag, configs=args, sampler=sampler)
    try:
        context_sem.read_train_valid_list()
        # change is_eval to False if running to cache features
        context_sem.iterate(is_eval=True)
    except Exception as e:
        with open('output/run_sem/sem_error.txt', 'a') as f:
            f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc(e))
