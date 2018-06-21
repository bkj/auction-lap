#!/usr/bin/env python

"""
    plot.py
"""

import json
import argparse
import pandas as pd

from rsub import *
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='results.jl')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    res = pd.DataFrame([json.loads(l) for l in open(args.inpath)])
    max_entry = res.max_entry[0]
    
    # --
    # Plot runtime
    
    _ = plt.plot(res.dim, res.auc_cpu_time, label='auc_cpu')
    _ = plt.plot(res.dim, res.auc_gpu_time, label='auc_gpu')
    _ = plt.plot(res.dim, res.gat_time, label='gat')
    _ = plt.plot(res.dim, res.src_time, label='src')
    _ = plt.legend()
    _ = plt.ylabel('seconds')
    _ = plt.xlabel('dim')
    _ = plt.title('Run time (max_entry=%d)' % max_entry)
    show_plot(outpath='time.png')
    
    # --
    # Plot accuracy
    
    mean_score = res[['auc_cpu_score', 'auc_gpu_score', 'gat_score', 'src_score']].values.mean(axis=-1)
    _ = plt.plot(res.dim, res.auc_cpu_score - mean_score, label='auc_cpu')
    _ = plt.plot(res.dim, res.auc_gpu_score - mean_score, label='auc_gpu')
    _ = plt.plot(res.dim, res.gat_score - mean_score, label='gat')
    _ = plt.plot(res.dim, res.src_score - mean_score, label='src')
    _ = plt.legend()
    _ = plt.ylabel('score (deviation from mean)')
    _ = plt.xlabel('dim')
    _ = plt.title('Score (max_entry=%d)' % max_entry)
    show_plot(outpath='score.png')