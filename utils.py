import numpy as np
import pandas as pd
import pickle, yaml
import os, datetime

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from math import atan2, degrees

from einops import rearrange

from tqdm import tqdm

def load_data(datadir):
    """
    load preprocessed data and subject metadata

    datadir: path containing pickled data

    returns
    point_data: array, sub x timepoint x feature
    info: pandas dataframe with video information
    """
    # point data
    trajectory_file = datadir + 'bigarray.pickle'
    assert trajectory_file, 'NameError: no trajectory data'

    # metadata
    metadata_file = datadir + 'bigarray_info_updated_13-10-22.pickle'
    assert metadata_file, 'NameError: no metadata'

    with open(trajectory_file, 'rb') as f:
        point_data = pickle.load(f)
        f.close()
    with open(metadata_file, 'rb') as f:
        info = pd.read_pickle(f)
        f.close()

    # add unique id to videos in same subject
    info['video'] = info.apply((lambda row: str(row.participant)+'_1' if row.timepoint==12 else str(row.participant)+'_2'), axis =1)

    # remove hi-res outlier
    print('removing 1 outlier video')
    keep_idx = info['video'] != '166064_1'
    point_data = point_data[:,:,keep_idx]
    info = info.loc[keep_idx,:].copy()

    # replace missing age at vid with timepoint
    index_all = ~(np.isnan(info['age_at_vid']))
    print('removing {:} videos with no age at video'.format((sum(1-index_all))))
    point_data = point_data[:,:,index_all]
    info = info.loc[index_all,:].copy()

    # rearrange point data to sub x T x feature
    point_data = np.transpose(point_data, axes=[2,0,1])

    assert len(point_data) == len(info)

    return point_data, info


def process_data(data, add_features=True, do_filter=True, filter_cutoff=5):
    """
    apply preprocessing to trajectory data

    :param data subject x time x (nodes x features) list of trajectory data arrays

    returns: processed data suject x time x (nodes x features) list of processed data arrays
    """

    r = rearrange(data, 'samples time (nodes features) -> samples nodes time features', nodes=18, features=2)

    processed_data = []

    for r0 in r:

        if add_features:
            d = rearrange(r0, 'nodes time features -> time (nodes features)')

            # joint angles
            ang = np.array([get_named_angles(r0[:,d,:]) for d in np.arange(np.shape(r0)[1])])
            # in radians
            ang = np.deg2rad(ang)

            # concatenate
            r0 = np.concatenate((d, ang), axis=-1)

        else:
            r0 = rearrange(r0, 'nodes time features -> time (nodes features)')

        if do_filter:
            sos = butter(4, filter_cutoff, 'low', fs=125, output='sos')
            r0 = sosfiltfilt(sos, r0, axis=0)

        # remove any nans *just in case*
        r0 = np.nan_to_num(r0, 0.0)

        processed_data.append(r0)

    return processed_data, []

def angle_between(p1, p2, p3):
    # https://stackoverflow.com/questions/58953047/issue-with-finding-angle-between-3-points-in-python
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360

    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def get_named_angles(data):
    angles = []
    # R shoulder -clk
    angles.append(angle_between(data[11,:], data[4,:], data[5,:]))
    #angles.append(min((angle_between(data[11,:], data[4,:], data[5,:]), angle_between(data[5,:], data[4,:], data[11,:]))))

    # R elbow - clk
    #angles.append(angle_between(data[6,:], data[5,:], data[4,:]))
    angles.append(min((angle_between(data[6,:], data[5,:], data[4,:]), angle_between(data[4,:], data[5,:], data[6,:]))))

    # R hip - anticlk
    #angles.append(angle_between(data[8,:], data[7,:], data[4,:]))
    angles.append(min((angle_between(data[8,:], data[7,:], data[4,:]), angle_between(data[4,:], data[7,:], data[8,:]))))

    # R knee - clk
    #angles.append(angle_between(data[7,:], data[8,:], data[9,:]))
    angles.append(min((angle_between(data[7,:], data[8,:], data[9,:]), angle_between(data[9,:], data[8,:], data[7,:]))))

    # R ankle - anticlk
    #angles.append(angle_between(data[10,:], data[9,:], data[8,:]))
    angles.append(min((angle_between(data[10,:], data[9,:], data[8,:]), angle_between(data[8,:], data[9,:], data[10,:]))))


    # L shoulder -anticlk
    angles.append(angle_between(data[12,:], data[11,:], data[4,:]))
    #angles.append(min((angle_between(data[12,:], data[11,:], data[4,:]), angle_between(data[4,:], data[11,:], data[12,:]))))

    # L elbow - anticlk
    #angles.append(angle_between(data[11,:], data[12,:], data[13,:]))
    angles.append(min((angle_between(data[11,:], data[12,:], data[13,:]), angle_between(data[13,:], data[12,:], data[11,:]))))

    # L hip - clk
    #angles.append(angle_between(data[11,:], data[14,:], data[15,:]))
    angles.append(min((angle_between(data[11,:], data[14,:], data[15,:]), angle_between(data[15,:], data[14,:], data[11,:]))))

    # L knee - anticlk
    #angles.append(angle_between(data[16,:], data[15,:], data[14,:]))
    angles.append(min((angle_between(data[16,:], data[15,:], data[14,:]), angle_between(data[14,:], data[15,:], data[16,:]))))

    # L ankle - clk
    #angles.append(angle_between(data[15,:], data[16,:], data[17,:]))
    angles.append(min((angle_between(data[15,:], data[16,:], data[17,:]), angle_between(data[17,:], data[16,:], data[15,:]))))

    return angles

def get_train_test_split(metadata, split = 0.33, random_state=None):
    """
    split data set into training (development) and testing (held-out) data.
    stratify on GMA score
    ensure subjects with multiple videos are in the same set

    metadata: sub x features

    returns:
    train_idx, test_idx: indices for train and test data
    """
    print('splitting data')
    # get first video for each participant
    unique_participants = metadata.drop_duplicates(subset = 'idnum', keep = 'first')

    X = unique_participants['idnum'].values[:,np.newaxis]
    Y = unique_participants['gma_vid_score'].values[:,np.newaxis]

    #train test split stratified by gma value
    x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size = split ,stratify = Y, random_state=random_state)

    # get all data for subjects in train
    train_index = np.where(metadata['idnum'].isin(x_train[:,0]))[0]
    test_index = np.where(metadata['idnum'].isin(x_validate[:,0]))[0]
    # no participants shared across groups
    assert len(set(metadata.iloc[train_index].participant.unique()) & set(metadata.iloc[test_index].participant.unique())) == 0

    return train_index, test_index

def get_params(config):
    """
    load parameters from configuration file and save to model directory
    :param config file, path to configuration
    :returns: params, dict of parameters
    """
    with open(config, 'rb') as f:
        params = yaml.safe_load(f)

    return params

# SVD
def run_svd(data, significance=None, n_perms=100):
    """
    :param data, n-obs by m-features array
    :param significance, 'mp' maximum eigenvalue of the Marchenko-Pastur distribution (assumes 0 mean and std=1)
                         'perm' calculates null ditribution of eigenvalues based on permutating columns independently
                          None, just return the SVD results
    :param n_perms, number of permutations to run, ignored if significance = None or 'mp'
    """
    
    # svd
    u, s, vh = np.linalg.svd(data, full_matrices=False)

    explained_variance_ratio = (s**2 / np.sum(s**2))

    eigenvectors = vh
    components = u

    if significance is None:
        return components, eigenvectors, s, explained_variance_ratio

    elif significance == 'mp':
        print('calculating theoretical maximum eigenvalue based on Marchenko-Pastur distribution')

        M,N = data.shape
        lambda_max = (1.0 + np.sqrt(N/M))**2

        lambda_max = (lambda_max * (M-1))**.5

        return components, eigenvectors, s, explained_variance_ratio, (s>lambda_max).astype(int), lambda_max

    elif significance == 'perm':
        print('running {:} permutations to estimate null distribution'.format(n_perms))
        perm_data = data.copy()
        lambda_max = 0

        for n in tqdm(np.arange(n_perms)):
            np.apply_along_axis(np.random.shuffle, axis=0, arr=perm_data)
            _, ps, _ = np.linalg.svd(perm_data, full_matrices=False)
            if ps[0] > lambda_max:
                lambda_max = ps[0]


        return components, eigenvectors, s, explained_variance_ratio,  (s>lambda_max).astype(int), lambda_max

    else:
        print('significance option not recognised, running SVD anyway')
        return components, eigenvectors, s, explained_variance_ratio
