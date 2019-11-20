"""@package filters
Filter function for stress strain data.
"""
import numpy as np
import scipy.signal
import pandas as pd


def load_data_set(files):
    """ Returns a list of stress-strain data.

    :param list files: Stress-strain data files to be loaded.
    :return list: (pd.DataFrame) Loaded data.

    Notes:
        - The header of the strain is 'e_true', the header of the stress is 'Sigma_true'
    """
    raw_tests = []
    for f in files:
        test_raw = pd.read_csv(f)
        raw_tests.append(test_raw)
    return raw_tests


def strain_filter(data, filter_type='sav_gol', window_length=11, poly_order=1):
    """ Returns a copy of the data with the filtered strain and stress.

    :param DataFrame data: true strain and true stress data, must contain a column of 'e_true'
    :param str filter_type: (optional) valid type of filter to use, default is Savitzy-Golay
    :param int window_length: window length in Sav-Gol filter, must be an odd integer
    :param int poly_order: order of polynomial to fit
    :return DataFrame: filtered true strain and true stress
    """

    strain = data['e_true']
    if filter_type == 'sav_gol':
        strain = scipy.signal.savgol_filter(strain, window_length, poly_order)

    data2 = pd.DataFrame(np.array([strain, data['Sigma_true']]).transpose(), columns=['e_true', 'Sigma_true'])
    return data2


def load_and_filter_data_set(files):
    """ Returns a list of the filtered data set in pandas DataFrame format.

    :param list files: [str] Paths to test data files, see the requirements for the files below.
    :return list: [pd.DataFrame] Collection of filtered data.

    Notes:
    - Each file has a column of true strain and true stress data
    - The header of the strain is 'e_true', the header of the stress is 'Sigma_true'
    """
    cleaned_tests = []
    for f in files:
        test_raw = pd.read_csv(f)
        test_filtered = strain_filter(test_raw)
        cleaned_tests.append(test_filtered)
    return cleaned_tests
