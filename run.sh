#!/bin/bash

# run.sh

# --
# Exact mode

python benchmark.py --max-entry 100 | tee results-100.jl
python plot.py --inpath results-100.jl

# --
# Approximate mode

python benchmark.py --max-entry 1000 --eps 100 | tee results-1000-approx.jl
python plot.py --inpath results-1000-approx.jl

# ^^ If you run this w/o the `eps` flag, auction will be _much_ slower