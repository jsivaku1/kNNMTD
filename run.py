import pandas as pd
import numpy as np
from kNNMTD import *
from utils import *

# Generate samples for unsupervised learning task
real = pd.read_csv('../Data/wisconsin_breast.csv')
model = kNNMTD(n_obs = 300, k=3, mode=-1)
synthetic = model.fit(real)
pcd = PCD(real,synthetic)

# Generate samples for classification task
real = pd.read_csv('../Data/cervical.csv')
model = kNNMTD(n_obs = 100, k=3, mode=0)
synthetic = model.fit(real,class_col='ca_cervix')
pcd = PCD(real,synthetic)

# Generate samples for regression task
real = pd.read_csv('/Users/jayanthsivakumar/Desktop/kNNMTD/Data/prostate.csv')
model = kNNMTD(n_obs = 100, k=4, mode=1)
synthetic = model.fit(real,class_col='lpsa')
pcd = PCD(real,synthetic)
print(pcd)