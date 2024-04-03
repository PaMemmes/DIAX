import collections
from typing import Any
from collections import defaultdict
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from utils.utils import DataSequence
from utils.utils import remove_infs, make_labels_binary, subset, encode

BATCH_SIZE = 256

FEATURES_DROPPED = [
    'Dst IP',
    'Flow ID',
    'Src IP',
    'Src Port',
    'Timestamp',
    'Bwd Seg Size Avg',
    'CWE Flag Count',
    'Bwd PSH Flags',
    'Fwd Seg Size Avg',
    'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg',
    'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg',
    'Bwd Blk Rate Avg',
    'Protocol',
    'Active Mean',
    'Pkt Len Min',
    'Fwd URG Flags',
    'Active Std',
    'Bwd Pkt Len Min',
    'Active Max',
    'Fwd PSH Flags',
    'Idle Std',
    'Fwd Pkt Len Min',
    'Bwd URG Flags',
    'Bwd Pkts/b Avg',
    'Pkt Len Var',
    'Bwd IAT Mean',
    'Flow Pkts/s',
    'Down/Up Ratio',
    'Active Min',
    'FIN Flag Cnt',
    'Pkt Size Avg']

# FEATURES_DROPPED = ['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp']

@dataclass
class DataFrame:
    filename: str = None

    df: pd.DataFrame() = None
    df_add: pd.DataFrame() = None

    df_cols: object = None

    le: LabelEncoder() = None

    x_test: object = None
    y_test: object = None


    train_sqc: Any = None
    test_sqc: Any = None

    test_frag_sqc: Any = None

    dfs: defaultdict() = None
    seperate_tests: defaultdict() = None

    def create_df(self, filename):
        if filename is not None:
            self.df = pd.read_csv(
                '/mnt/md0/files/cicids2018/' +
                filename,
                engine='python')
        else:
            all_files = glob.glob(
                os.path.join(
                    '/mnt/md0/files/cicids2018/',
                    "*.csv"))
            self.df = pd.concat((pd.read_csv(f, engine='python')
                                for f in all_files), ignore_index=True)

        print('Length of CSE-CICIDS2018 data', len(self.df))
        self.df = self.df.sample(frac=1)
        self.df = self.df.drop(FEATURES_DROPPED, axis=1)
        self.df_cols = self.df.columns

    def create_label_encoder(self):
        labels = self.df['Label']
        self.le = LabelEncoder()
        self.le.fit(labels)

    def create_oos_test(self, test_size):
        # Make-out-of-sample test split, s.t. additional data is not
        # incorporated
        self.df = self.df.reset_index(drop=True)
        df, labels = remove_infs(self.df)
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        _, self.x_test, y_train, self.y_test = train_test_split(
            df, labels, test_size=test_size, shuffle=False)
        self.df = self.df[:int((1 - test_size) * len(self.df))]

        normals = collections.Counter(y_train)[0]
        anomalies = collections.Counter(y_train)[1]
        self.anomalies_percentage = anomalies / (normals + anomalies)

    def preprocess_add(self, add_data):
        x = np.array(add_data)
        x = np.array(x).reshape((x.shape[0] * x.shape[1]), x.shape[2])
        y = np.array(['ANOMALY' for i in range(len(x))]).reshape(len(x), 1)
        data = np.concatenate((x, y), axis=1)
        self.df_add = pd.DataFrame(data, columns=self.df_cols)

    def preprocess(
            self,
            filename=None,
            kind=None,
            add=None,
            test_size=0.15,
            scale=True):
        self.create_df(filename)
        self.create_label_encoder()
        self.create_oos_test(test_size)
    
        if add is not None:
            self.df = pd.concat([self.df_add, self.df], ignore_index=True)
            self.df = self.df.sample(frac=1)
        print('Length of df (w/wo frags and add) before removing infs', len(self.df))
        df, labels = remove_infs(self.df)
        print('Length of df (w/wo frags and add) after removing infs', len(df))
        labels = encode(self.le, labels)
        labels = make_labels_binary(self.le, labels)
        x_train, _, y_train, _ = train_test_split(
            df, labels, test_size=test_size, shuffle=False)

        # Subsetting only Normal Network packets in training set
        if kind == 'normal':
            x_train, y_train = subset(x_train, y_train, 0)
        elif kind == 'anomaly':
            x_train, y_train = subset(x_train, y_train, 1)
            print('Using only anomaly data')

        if scale is True:
            scaler = MinMaxScaler()

            x_train = scaler.fit_transform(x_train)
            self.x_test = scaler.transform(self.x_test)

        self.x_test = self.x_test

        self.train_sqc = DataSequence(x_train, y_train, batch_size=BATCH_SIZE)
        self.test_sqc = DataSequence(
            self.x_test, self.y_test, batch_size=BATCH_SIZE)


    def seperate_dfs(self, filename, test_size=0.15):
        if filename is not None:
            df_all = pd.read_csv(
                '/mnt/md0/files/cicids2018/' +
                filename,
                engine='python')
        else:
            all_files = glob.glob(
                os.path.join(
                    '/mnt/md0/files/cicids2018',
                    "*.csv"))
            df_all = pd.concat((pd.read_csv(f, engine='python')
                               for f in all_files), ignore_index=True)

        _df = df_all.copy()
        _df = _df.drop(FEATURES_DROPPED, axis=1)
        _df = _df.reset_index(drop=True)
        _df, _labels = remove_infs(_df)
        _x_test = _df.to_numpy()
        scaler = MinMaxScaler()
        scaler.fit(_x_test)

        dfs = defaultdict()
        self.seperate_tests = defaultdict()
        for col in df_all['Label'].unique():
            dfs[col] = df_all[df_all['Label'] == col]
            df = dfs[col].sample(frac=1)
            if len(df) <= 5:
                continue
            df = df.drop(FEATURES_DROPPED, axis=1)
            df, y_test = remove_infs(df)
            y_test = encode(self.le, y_test)
            x_test = df.to_numpy()

            test_sqc = DataSequence(x_test, y_test, batch_size=BATCH_SIZE)
            self.seperate_tests[col] = test_sqc
