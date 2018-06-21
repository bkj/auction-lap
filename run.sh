#!/bin/bash

# run.sh

python benchmark.py --max-entry 100 | tee results-100.jl
python plot.py --inpath results-100.jl

python benchmark.py --max-entry 1000 --eps 1 | tee results-1000.jl
python plot.py --inpath results-1000.jl