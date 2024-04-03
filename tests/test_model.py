import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.utils import make_labels_binary, subset, save_results

@pytest.fixture
def example_labels():
    return np.array(['Benign', 'Benign', 'BOT', 'DDoS', 'Trojan', 'Worm', 'Scan', 'Benign', 'Benign', 'Trojan', 'Worm', 'Scan'])

@pytest.fixture
def example_le(example_labels):
    le = LabelEncoder()
    labels = example_labels
    le.fit(labels)
    int_labels = le.transform(labels)

    return le, int_labels

@pytest.fixture
def example_df_labels():
    return pd.DataFrame({'Label': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]})

@pytest.fixture
def example_df():
    return pd.DataFrame({'Feature1': [1, 3, 5, float('Nan'), 100000, 4, -10, 4, 5, float('inf'), float('-inf'), float('nan')],
                         'Feature2': [3, 12, 5, 3, 3, 1, 4, 10, 51, 5, 10000, -1053445]})

def test_make_labels_binary(example_le):
    le, int_labels = example_le
    new_labels = make_labels_binary(le, int_labels)
    unique = np.unique(new_labels)
    assert isinstance(new_labels, np.ndarray)
    assert len(unique) == len([0,1])
    assert all([a == b for a, b in zip(unique, [0,1])])

def test_subset_normal(example_df, example_df_labels):
    x_train = example_df
    y_train = example_df_labels
    x_train, y_train = subset(x_train, y_train)
    assert len(x_train) == 5
    assert len(y_train) == 5
    
