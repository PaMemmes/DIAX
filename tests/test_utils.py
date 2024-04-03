import pytest
import pandas as pd
import numpy as np
from src.utils.utils import remove_infs, encode, subset, make_labels_binary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

@pytest.fixture
def example_df():
    return pd.DataFrame({'Feature1': [1, 3, 5, float('Nan'), 100000, 4, -10, 4, 5, float('inf'), float('-inf'), float('nan')],
                         'Feature2': [3, 12, 5, 3, 3, 1, 4, 10, 51, 5, 10000, -1053445],
                         'Label': ['Benign', 'Benign', 'BOT', 'DDoS', 'Trojan', 'Worm', 'Scan', 'Benign', 'Benign', 'Trojan', 'Worm', 'Scan']})

@pytest.fixture
def example_xy(example_df):
    le = LabelEncoder()
    le.fit(example_df['Label'])
    labels = encode(le, example_df['Label'])
    df, labels = remove_infs(example_df)
    labels = make_labels_binary(le, labels)
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)
    return x_train, y_train

@pytest.fixture
def example_le(example_df):
    le = LabelEncoder()
    le.fit(example_df['Label'])
    labels = encode(le, example_df['Label'])
    return le, labels

def test_remove_infs(example_df):
    df, labels = remove_infs(example_df)
    assert df.isnull().sum().sum() == 0
    assert np.isinf(df).sum().sum() == 0

def test_encode(example_df):
    le = LabelEncoder()
    le.fit(example_df['Label'])
    labels = encode(le, example_df['Label'])
    for elem in labels:
        assert isinstance(elem, np.int64)
    assert len(np.unique(labels)) == 6

def test_subset(example_xy):
    x_train, y_train = example_xy
    x_0, y_0 = subset(x_train, y_train, 0)
    x_1, y_1 = subset(x_train, y_train, 1)
    assert len(x_0) == len(y_0)
    assert len(x_1) == len(y_1)
    for elem in y_0:
        assert elem == 0
    for elem in y_1:
        assert elem == 1

def test_make_labels_binary(example_le):
    le, labels = example_le
    new_labels = make_labels_binary(le, labels)
    assert isinstance(new_labels, np.ndarray)
    assert len(labels) == len(new_labels)
    for elem in new_labels:
        assert elem == 0 or elem == 1
