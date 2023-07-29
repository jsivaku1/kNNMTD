# Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors

## Overview
The repository implements the synthetic sampling technique called kNNMTD using the small datasets. The algorithm uses a three-step procedure. 
- Firstly, the k-Nearest Neighbor (kNN) algorithm is applied on each of the instances
- Secondly, the neighboring samples are diffused using mega-trend diffusion (MTD)
- Finally, the samples are generated using the domain ranges from MTD through plausibility assessment mechanism, then kNN is applied on the synthetic samples to select the closest acceptable samples 

The following illustration show how the algorithm generates artificial samples. For more information, refer the original [paper](https://doi.org/10.1016/j.knosys.2021.107687).
<div align="left">
<br/>
<p align="center">
<img align="center" width=90% src="https://github.com/jsivaku1/kNNMTD/blob/main/illustration.png"></img>
</p>
</div>

All the final benchmark datasets used in the paper after preprocessing is available inside the data folder.

- mode = -1 &#8594; Unsupervised 
- mode = 0 &#8594; Classification
- mode = 1 &#8594; Regression


## Usage 
```python3
import pandas as pd
import numpy as np
from kNNMTD import *
from utils import *

# Generate samples for unsupervised learning task
real = pd.read_csv('../Data/wisconsin_breast.csv')
model = kNNMTD(n_obs = 300,k=3,mode=-1)
synthetic = model.fit(real)
pcd = PCD(real,synthetic)

# Generate samples for classification task
real = pd.read_csv('../Data/cervical.csv')
model = kNNMTD(n_obs = 100,k=3,mode=0)
synthetic = model.fit(real,class_col='ca_cervix')
pcd = PCD(real,synthetic)

# Generate samples for regression task
real = pd.read_csv('../Data/prostate.csv')
model = kNNMTD(n_obs = 100,k=4,mode=1)
synthetic = model.fit(real,class_col='lpsa')
pcd = PCD(real,synthetic)
```

# Citing kNNMTD

Please cite the following work if you are using the source code:

- Jayanth Sivakumar, Karthik Ramamurthy, Menaka Radhakrishnan, and Daehan Won. "Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors." Knowledge-Based Systems (2021): 107687.

```LaTeX
@article{sivakumar2021synthetic,
  title={Synthetic sampling from small datasets: A modified mega-trend diffusion approach using k-nearest neighbors},
  author={Sivakumar, Jayanth and Ramamurthy, Karthik and Radhakrishnan, Menaka and Won, Daehan},
  journal={Knowledge-Based Systems},
  pages={107687},
  year={2021},
  publisher={Elsevier}
}
```
