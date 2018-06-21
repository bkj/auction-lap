#!/usr/bin/env python

"""
    benchmark.py
"""

from __future__ import print_function, division

import sys
import json
import torch
import argparse
import numpy as np
from time import time

from lap import lapjv as jv_gat   # gatagat
from lapjv import lapjv as jv_src # src-d
from auction_lap import auction_lap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-entry', type=int, default=100, help='maximum entry in matrix')
    parser.add_argument('--min-dim', type=int, default=1000, help='minimum dimension matrix to test')
    parser.add_argument('--max-dim', type=int, default=10000, help='maximum dimension matrix to test')
    parser.add_argument('--n-evals', type=int, default=10, help='number of steps between min and max matrix size')
    parser.add_argument('--eps', type=int, help='"bid size" -- smaller values give better accuracy w/ longer runtime')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    np.random.seed(args.seed)
    
    for dim in np.linspace(args.min_dim, args.max_dim, args.n_evals, dtype=int):
        
        X = np.random.choice(args.max_entry, (dim, dim))
        
        t = time()
        _, gat_ass, _ = jv_gat(X.max() - X)
        gat_score = X[(np.arange(X.shape[0]), gat_ass)].sum()
        gat_time = time() - t
        
        t = time()
        src_ass, _, _ = jv_src(X.max() - X)
        src_score = X[(np.arange(X.shape[0]), src_ass)].sum()
        src_time = time() - t
        
        # Run auction solver
        Xt_cpu = torch.from_numpy(X).float()
        Xt_gpu = Xt_cpu.cuda()
        
        t = time()
        auc_cpu_score, auc_cpu_ass = auction_lap(Xt_cpu, eps=args.eps) # Score is accurate to within n * eps
        auc_cpu_time = time() - t
        
        t = time()
        auc_gpu_score, auc_gpu_ass = auction_lap(Xt_gpu, eps=args.eps) # Score is accurate to within n * eps
        auc_gpu_time = time() - t
        
        print(json.dumps({
                "max_entry"     : int(args.max_entry),
                "dim"           : int(dim),
                
                "gat_score"     : int(gat_score),
                "src_score"     : int(src_score),
                "auc_cpu_score" : int(auc_cpu_score),
                "auc_gpu_score" : int(auc_gpu_score),
                
                "gat_time"      : float(gat_time),
                "src_time"      : float(src_time),
                "auc_cpu_time"  : float(auc_cpu_time),
                "auc_gpu_time"  : float(auc_gpu_time),
        }))
        sys.stdout.flush()
