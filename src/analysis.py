import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from itertools import combinations
from ydata_profiling import ProfileReport

from utils.utils import remove_infs

def bar_plot_agg(df):

    df['Label'] = df['Label'].replace(
        {
            'DoS-attacks-SlowHTTPTest': 'DoS',
            'DoS attacks-Slowloris': 'DoS',
            'DoS attacks-Hulk': 'DoS'},
        regex=True)
    df['Label'] = df['Label'].replace(
        {
            'DDoS attack-LOIC-UDP': 'DDoS',
            'DDoS attack-HOIC': 'DDoS',
            'DDoS attacks-LOIC-HTTP': 'DDoS'},
        regex=True)
    df['Label'] = df['Label'].replace(
        {
            'Brute Force -Web': 'Brute Force',
            'Brute Force -XSS': 'Brute Force',
            'SSH-BruteForce': 'Brute Force',
            'FTP:BruteForce': 'Brute Force'},
        regex=True)

    label_counts = df['Label'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(label_counts)))
    ax.set_xticklabels(label_counts.index.tolist())
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Percentage')
    bar = ax.bar(x=label_counts.index.tolist(), height=label_counts)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    ax.bar_label(ax.containers[0], fmt='%.4f')
    fig = ax.get_figure()
    fig.savefig(
        "../analysis_plots/distribution_cicids2018_agg.pdf",
        bbox_inches="tight")
    plt.close()


def bar_plot_binary(df):

    subset_benign = df[df['Label'] == 'Benign']
    subset_anomaly = df[df['Label'] != 'Benign']

    percentage_benign = len(subset_benign)
    percentage_anomaly = len(subset_anomaly)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Benign', 'Anomaly'])
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Percentage')

    bar = ax.bar(
        x=np.arange(2),
        height=[
            percentage_benign,
            percentage_anomaly])
    plt.tight_layout()
    ax.bar_label(bar, fmt='%.3f')
    fig = ax.get_figure()
    fig.savefig(
        "../analysis_plots/distribution_cicids2018_binary.pdf",
        bbox_inches="tight")
    plt.close()


def bar_plot(df):
    label_counts = df['Label'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(len(label_counts)))
    ax.set_xticklabels(label_counts.index.tolist())
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Percentage')
    bar = ax.bar(label_counts.index.tolist(), height=label_counts)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    ax.set(xlabel='Attack Type', ylabel='Percentage')
    ax.bar_label(ax.containers[0], fmt='%.4f')
    fig = ax.get_figure()
    fig.savefig(
        "../analysis_plots/distribution_cicids2018.pdf",
        bbox_inches="tight")
    plt.close()


def reduce_corr(df, threshold):
    under_thresh = set()
    for (row, column) in combinations(df.columns, 2):
        if (abs(df.loc[row, column]) >= threshold):
            under_thresh.add(row)
            under_thresh.add(column)
    under_thresh = sorted(under_thresh)
    new_corr = df.loc[under_thresh, under_thresh]
    return new_corr


def plot_corr(df):
    corr = df.corr()
    plt.figure(figsize=(19, 10))
    corr = reduce_corr(corr, 0.9)
    sns.heatmap(corr, cmap="YlGnBu")
    plt.savefig('../analysis_plots/corr.pdf')
    plt.show()


def plot_dists(df):
    for col in df.columns:
        save = col.replace("/", "_").strip()
        sns.kdeplot(
            x=pd.to_numeric(
                df[col]), fill=True, log_scale=(
                False, True))
        plt.savefig('../analysis_plots/' + save + '_kde_log.pdf')
        plt.close()
    for col in df.columns:
        save = col.replace("/", "_").strip()
        sns.kdeplot(x=pd.to_numeric(df[col]), fill=True)
        plt.savefig('../analysis_plots/' + save + '_kde.pdf')
        plt.close()


if __name__ == '__main__':
    all_files = glob.glob(
        os.path.join(
            '/mnt/md0/files_memmesheimer/cicids2018',
            "*.csv"))
    df = pd.concat((pd.read_csv(f, engine='python')
                   for f in all_files), ignore_index=True)
    df = df.drop(['Dst IP', 'Flow ID', 'Src IP',
                 'Src Port', 'Timestamp'], axis=1)
    df, _ = remove_infs(df)
    plot_corr(df)
    bar_plot_binary(df)
    bar_plot(df)
    bar_plot_agg(df)
    plot_dists(df)
