import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score,\
                                roc_curve, roc_auc_score, precision_recall_curve,auc, average_precision_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import random
from scipy.spatial import distance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss 
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import heapq

class kNNMTD():
    # type = 0 => Classification
    # type = 1 => Regression
    # type = -1 => Unsupervised 
    def __init__(self, n_obs=100, k=3,mode=-1, random_state=1):
        self.type = mode
        self.n_obs = n_obs
        self._gen_obs = self.n_obs *10
        self.k = k
        np.random.RandomState(random_state)

        
    def diffusion(self, sample):
        if(type(sample.values[0] == str)):
            sample = np.array(sample,dtype=float)
        new_sample = []
        n = len(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2
        if(u_set == min_val or u_set == max_val):
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample,ddof=1)
        if(var == 0):
            a = min_val/5
            b = max_val*5
            h=0
            new_sample = np.random.uniform(a, b, size=self._gen_obs) 
        else:
            h = var / n
            a = u_set - (skew_l * np.sqrt(-2 * (var/Nl) * np.log(10**(-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var/Nu) * np.log(10**(-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            while(len(new_sample) < self._gen_obs):
                    x = np.random.uniform(L,U)
                    if(x <= u_set):
                        MF = (x-L) / (u_set-L)
                    elif(x > u_set):
                        MF = (U-x)/(U-u_set)
                    elif(x < L or x > U) :
                        MF = 0
                    rs = np.random.uniform(0,1)
                    if(MF > rs):
                        new_sample.append(x)
                    else:
                        continue
        return np.array(new_sample)

    def getNeighbors(self, val_to_test, xtrain, ytrain):
        if(self.type == 0):
            X_train = xtrain.reshape(-1, 1)
            y_train = ytrain
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            dist, nn_indices = knn.kneighbors(X=np.array(val_to_test).reshape(1,-1), return_distance=True)
            neighbor_df = xtrain[np.squeeze(nn_indices)]
        elif(self.type == 1):
            X_train = xtrain.reshape(-1, 1)
            y_train = ytrain
            knn = KNeighborsRegressor(n_neighbors=3)
            knn.fit(X_train, y_train)
            dist, nn_indices = knn.kneighbors(X=np.array(val_to_test).reshape(1,-1), return_distance=True)
            neighbor_df = xtrain[np.squeeze(nn_indices)]
        else:
            X_train = xtrain.reshape(-1, 1)
            y_train = ytrain
            dist = [np.square(x - val_to_test) for x in X_train]
            nn_indices = heapq.nsmallest(3, range(len(dist)), dist.__getitem__)
            neighbor_df = xtrain[np.squeeze(nn_indices)]
        return nn_indices, neighbor_df

    def fit(self, train,class_col=None):
        train.reset_index(inplace=True, drop=True)
        if(self.type != -1):
            X_train = train.drop(class_col,axis=1)
            y_train = train[class_col]
        else:
            X_train = train.copy()
        columns = X_train.columns
        temp_surr_data = pd.DataFrame(columns = list(train.columns))
        surrogate_data = pd.DataFrame(columns = list(train.columns))
        synth_data = pd.DataFrame(columns = list(train.columns))
        temp = pd.DataFrame(columns = list(train.columns))
        if(self.type == 0):
            for t in np.unique(train[class_col]):
                train_class_df = train[train[class_col] == t]
                train_class_df.reset_index(inplace=True, drop=True)
                for ix,val in train_class_df.iterrows():
                    for col in columns:
                        X_train = train_class_df[col].values.reshape(-1, 1)
                        y_train = train_class_df[class_col].values
                        knn = KNeighborsClassifier(n_neighbors=self.k)
                        knn.fit(X_train, y_train)
                        nn_indices = knn.kneighbors(X=np.array(val[col]).reshape(1,-1), return_distance=False)
                        neighbor_df = train_class_df[col][np.squeeze(nn_indices)]
                        if(col != class_col and (neighbor_df.dtype == np.int64 or neighbor_df.dtype == np.int32)): 
                            bin_val =  np.unique(train_class_df[col])
                            centers = (bin_val[1:]+bin_val[:-1])/2
                            x = self.diffusion(neighbor_df)  
                            ind = np.digitize(x, bins=centers , right=True)
                            x = np.array([bin_val[i] for i in ind])
                            y = np.array([t for _ in range(self._gen_obs)])
                            nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, y)
                            temp[col] = pd.Series(neighbor_val_array)  
                        elif(col==class_col):
                            temp[col] = pd.Series(np.array([t for _ in range(self._gen_obs)]))
                        else:
                            x = self.diffusion(neighbor_df)
                            y = np.array([t for _ in range(self._gen_obs)])
                            nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, y)
                            temp[col] = pd.Series(neighbor_val_array)        
                    temp[class_col] = t
                    temp_surr_data = pd.concat([temp_surr_data, temp])            
            surrogate_data = pd.concat([surrogate_data, temp_surr_data])                          
            synth_data = self.sample(surrogate_data,train,class_col)     
        elif(self.type == 1):
            train_class_df = train.copy()
            train_class_df.reset_index(inplace=True, drop=True)
            for ix,val in train_class_df.iterrows():
                for col in columns:
                    X_train = train_class_df[col].values.reshape(-1, 1)
                    y_train = train_class_df[class_col].values
                    knn = KNeighborsRegressor(n_neighbors=self.k)
                    knn.fit(X_train, y_train)
                    #Find corresponding attribute neighbors
                    nn_indices = knn.kneighbors(X=np.array(val[col]).reshape(1,-1), return_distance=False)
                    neighbor_df = train_class_df[col][np.squeeze(nn_indices)]    
                    y_neighbor_df = train_class_df[class_col][np.squeeze(nn_indices)]
                    if(neighbor_df.dtype == np.int64 or neighbor_df.dtype == np.int32): 
                        bin_val =  np.unique(train_class_df[col])
                        centers = (bin_val[1:]+bin_val[:-1])/2
                        x = self.diffusion(neighbor_df)  
                        ind = np.digitize(x, bins=centers , right=True)
                        x = np.array([bin_val[i] for i in ind])
                        y = self.diffusion(y_neighbor_df)  
                        nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, y)
                        temp[col] = pd.Series(neighbor_val_array)  
                    else:
                        x = self.diffusion(neighbor_df)
                        y = self.diffusion(y_neighbor_df)
                        nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, y)
                        temp[col] = pd.Series(neighbor_val_array)        
                temp_surr_data = pd.concat([temp_surr_data, temp])    
            surrogate_data = pd.concat([surrogate_data, temp_surr_data])           
            synth_data = self.sample(surrogate_data,train,class_col)     
        else:
            train_class_df = train.copy()
            train_class_df.reset_index(inplace=True, drop=True)
            for ix,val in train_class_df.iterrows():
                for col in columns:
                    X_train = train_class_df[col].values.reshape(-1, 1)
                    dist = [np.square(x - val[col]) for x in X_train]
                    nn_indices = heapq.nsmallest(self.k, range(len(dist)), dist.__getitem__)
                    neighbor_df = train_class_df[col][np.squeeze(nn_indices)]    
                    if(neighbor_df.dtype == np.int64 or neighbor_df.dtype == np.int32): 
                        bin_val =  np.unique(train_class_df[col])
                        centers = (bin_val[1:]+bin_val[:-1])/2
                        x = self.diffusion(neighbor_df)  
                        ind = np.digitize(x, bins=centers , right=True)
                        x = np.array([bin_val[i] for i in ind])
                        nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, None)
                        temp[col] = pd.Series(neighbor_val_array)  
                    else:
                        x = self.diffusion(neighbor_df)
                        nn_indices, neighbor_val_array = self.getNeighbors(val[col], x, None)
                        temp[col] = pd.Series(neighbor_val_array)        
                temp_surr_data = pd.concat([temp_surr_data, temp])    
            surrogate_data = pd.concat([surrogate_data, temp_surr_data])           
            synth_data = self.sample(surrogate_data,train,class_col=None)             
        return synth_data

    def sample(self, data,real,class_col):
        surrogate_data = data.copy()
        train = real.copy()
        synth_data = pd.DataFrame(columns = list(train.columns))
        for x in train.columns:
            surrogate_data[x]=surrogate_data[x].astype(train[x].dtypes.name)
        surrogate_data.reset_index(inplace=True, drop=True)
        if(not isinstance(class_col,type(None)) and len(np.unique(train[class_col]))<=5):
            num, div = np.abs(self.n_obs), len(np.unique(surrogate_data[class_col]))
            class_num = [num // div + (1 if x < num % div else 0)  for x in range (div)]
            for i in range(len(np.unique(surrogate_data[class_col]))):
                try:
                    temp_data = surrogate_data[surrogate_data[class_col] == np.unique(surrogate_data[class_col])[i]].sample(class_num[i])
                except ValueError:
                    temp_data = surrogate_data[surrogate_data[class_col] == np.unique(surrogate_data[class_col])[i]].sample(class_num[i], replace='True')
                synth_data = pd.concat([synth_data,temp_data],axis=0)
        else:
            try:
                temp_data = surrogate_data.sample(np.abs(self.n_obs))
            except ValueError:
                temp_data = surrogate_data.sample(np.abs(self.n_obs), replace='True')
            synth_data = pd.concat([synth_data,temp_data],axis=0) 
        for x in train.columns:
            synth_data[x]=synth_data[x].astype(train[x].dtypes.name)
        synth_data.reset_index(inplace=True, drop=True)
        return synth_data