import numpy as np
import pandas as pd
import pickle, yaml
import os, datetime

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, resample, hilbert
from scipy.interpolate import interp1d

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


def process_data(data, new_frequency = 10, do_filter=True, frequency=25):
    """
    apply preprocessing to trajectory data

    :param data subject x time x (nodes x features) list of trajectory data arrays

    returns: processed data suject x time x (nodes x features) list of processed data arrays
    """

    processed_data = []
    processed_outliers = []

    if do_filter:
        sos = butter(4, (0.01, new_frequency // 2), 'bandpass', fs=frequency, output='sos')
        data = sosfiltfilt(sos, data, axis=1)

    # outlier points
    r_data = rearrange(data, 's t f -> (s t) f')
    median_position = np.median(r_data, axis=0)
    mad = np.median(abs(r_data - median_position), axis=0)
    mad_threshold = 3 * mad * 1.4826
    outliers = (abs(r_data - median_position) > mad_threshold)
    outliers = rearrange(outliers, '(s t) f -> s t f', s=len(data))

    if new_frequency != frequency:

        for n, r0 in enumerate(data):

            o0 = outliers[n]
            # interpolate to lower frequency, accountinng for outlier regions
            npoints, nfeatures = np.shape(r0)
            new_npoints = int(npoints *  (new_frequency / frequency))

            new_r0 = np.zeros((new_npoints, nfeatures))
            for f in np.arange(nfeatures):
                r0_feature = r0[:, f]
                keep_points = ~o0[:, f]
                # interpolate (ignoring outliers)
                interpf = interp1d(np.arange(npoints)[keep_points], r0_feature[keep_points], kind='slinear', bounds_error=False, fill_value = r0_feature[keep_points][-1])
                new_r0_feature = interpf(np.linspace(0, npoints, new_npoints))
                new_r0[:,f] = new_r0_feature
            r0 = new_r0.copy()

            # remove any nans *just in case*
            r0 = np.nan_to_num(r0, 0.0)

            processed_data.append(r0)
    else:
        processed_data = [data[n,:,:] for n in np.arange(len(data))]

    return np.array(processed_data), outliers

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
