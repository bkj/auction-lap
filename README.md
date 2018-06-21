#### auction-lap

Linear Assignment Problem (LAP) solver using the auction algorithm.  Implemented in `pytorch`, runs on CPU or GPU.

##### Installation

```
conda create -n auction_env python=3.6 pip
source activate auction_env
pip install -r requirements.txt
conda install pytorch==0.3.1 torchvision cuda91 -c pytorch -y
```

##### Usage

See `./run.sh` for usage

##### To Do

 - Would this give any speedups?
    - https://github.com/rusty1s/pytorch_scatter