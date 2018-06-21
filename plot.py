#!/usr/bin/env python

"""
    plot.py
"""

from __future__ import print_function

import sys
import json
import argparse
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='results.jl')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    res = pd.DataFrame([json.loads(l) for l in open(args.inpath)])
    max_entry = res.max_entry[0]
    eps       = res.eps[0]
    
    # --
    # Plot runtime
    
    # _ = plt.plot(res.dim, res.auc_cpu_time, label='auc_cpu')
    _ = plt.plot(res.dim, res.auc_gpu_time, label='auc_gpu')
    _ = plt.plot(res.dim, res.gat_time, label='gat')
    _ = plt.plot(res.dim, res.src_time, label='src')
    _ = plt.legend()
    _ = plt.ylabel('seconds')
    _ = plt.xlabel('dim')
    _ = plt.title('Run time (max_entry=%d | eps=%s)' % (max_entry, str(eps)))
    print('plot.py: saving time.png', file=sys.stderr)
    plt.savefig('time.png')
    plt.close()
    
    # --
    # Plot accuracy
    
    best_score = res.gat_score.values
    # _ = plt.plot(res.dim, 1 - res.auc_cpu_score / best_score, label='auc_cpu')
    _ = plt.plot(res.dim, 1 - res.auc_gpu_score / best_score, label='auc_gpu')
    _ = plt.plot(res.dim, 1 - res.gat_score / best_score, label='gat')
    _ = plt.plot(res.dim, 1 - res.src_score / best_score, label='src')
    _ = plt.legend()
    _ = plt.ylabel('error (score / best_score)')
    _ = plt.xlabel('dim')
    _ = plt.title('Score (max_entry=%d | eps=%s)' % (max_entry, str(eps)))
    print('plot.py: saving score.png', file=sys.stderr)
    plt.savefig('score.png')
    plt.close()