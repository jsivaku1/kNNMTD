import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class kNNMTD():
    """
    k-Nearest Neighbor Mega-Trend Diffusion (kNNMTD).
    This is a corrected and optimized version that follows the iterative logic of the
    original paper but uses vectorization to significantly speed up the process.
    """
    def __init__(self, n_obs=100, k=5, random_state=None, n_epochs=10):
        if k < 2:
            raise ValueError("k must be at least 2.")
        self.n_obs = n_obs
        self.k = k
        self.rng = np.random.default_rng(random_state)
        self.n_epochs = n_epochs

    def _diffusion(self, sample_array):
        """Calculates diffusion bounds for a 2D array of neighbor samples."""
        min_vals = np.min(sample_array, axis=1)
        max_vals = np.max(sample_array, axis=1)
        u_set = (min_vals + max_vals) / 2.0
        
        N_L = np.sum(sample_array < u_set[:, np.newaxis], axis=1)
        N_U = np.sum(sample_array >= u_set[:, np.newaxis], axis=1)
        
        total_N = N_L + N_U
        total_N[total_N == 0] = 1
        
        skew_L = N_L / total_N
        skew_U = N_U / total_N
        
        variance = np.var(sample_array, axis=1, ddof=1)
        variance[np.isnan(variance)] = 0
        
        a = np.zeros_like(variance)
        b = np.zeros_like(variance)
        
        zero_var_mask = (variance == 0)
        a[zero_var_mask] = min_vals[zero_var_mask] / 5.0
        b[zero_var_mask] = max_vals[zero_var_mask] * 5.0
        
        non_zero_mask = ~zero_var_mask
        safe_N_L = np.copy(N_L[non_zero_mask]); safe_N_L[safe_N_L == 0] = 1
        safe_N_U = np.copy(N_U[non_zero_mask]); safe_N_U[safe_N_U == 0] = 1
        
        log_term = np.log(10**-20)
        sqrt_term_L = np.sqrt(-2 * (variance[non_zero_mask] / safe_N_L) * log_term)
        sqrt_term_U = np.sqrt(-2 * (variance[non_zero_mask] / safe_N_U) * log_term)
        
        a[non_zero_mask] = u_set[non_zero_mask] - skew_L[non_zero_mask] * sqrt_term_L
        b[non_zero_mask] = u_set[non_zero_mask] + skew_U[non_zero_mask] * sqrt_term_U
        
        return np.minimum(a, min_vals), np.maximum(b, max_vals)

    def _get_pam_samples(self, lb, ub, u_set, pool_size):
        """Generates samples using PAM in a vectorized way."""
        # Shape of lb, ub, u_set is (n_features,)
        # We generate candidates for each feature independently
        candidate_samples = self.rng.uniform(lb, ub, size=(pool_size, len(lb)))
        
        mf_vals = np.zeros_like(candidate_samples)
        
        # Calculate MF for each feature's candidates
        for i in range(len(lb)):
            col_candidates = candidate_samples[:, i]
            
            mask1 = col_candidates <= u_set[i]
            denom1 = u_set[i] - lb[i]
            if denom1 != 0: mf_vals[mask1, i] = (col_candidates[mask1] - lb[i]) / denom1
            
            mask2 = col_candidates > u_set[i]
            denom2 = ub[i] - u_set[i]
            if denom2 != 0: mf_vals[mask2, i] = (ub[i] - col_candidates[mask2]) / denom2
            
        mf_product = np.prod(mf_vals, axis=1)
        random_thresholds = self.rng.uniform(0, 1, size=pool_size)
        return candidate_samples[mf_product > random_thresholds]

    def fit_generate(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        task_mode = 'unsupervised'
        if y is not None:
            task_mode = 'regression' if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15 else 'classification'

        full_real_df = pd.concat([X, y.to_frame(name=y.name)], axis=1) if y is not None else X
        
        all_synthetic_data = []
        obs_per_epoch = int(np.ceil(self.n_obs / self.n_epochs))

        # Pre-fit nearest neighbor models to speed up the loop
        nn_models = {}
        if task_mode == 'classification':
            for class_label in y.unique():
                class_data = full_real_df[y == class_label]
                if len(class_data) >= self.k:
                    # Use .values to avoid feature name warnings
                    nn_models[class_label] = NearestNeighbors(n_neighbors=self.k).fit(class_data.values)
        else:
            nn_models['all'] = NearestNeighbors(n_neighbors=self.k).fit(full_real_df.values)

        for epoch in range(self.n_epochs):
            surrogate_rows = []
            for i in range(len(full_real_df)):
                real_row_df = full_real_df.iloc[[i]]
                
                model, search_data = None, full_real_df
                if task_mode == 'classification':
                    current_class = y.iloc[i]
                    if current_class in nn_models:
                        model = nn_models[current_class]
                        search_data = full_real_df[y == current_class]
                else:
                    model = nn_models.get('all')
                
                if model is None: continue

                _, indices = model.kneighbors(real_row_df.values)
                neighbor_df = search_data.iloc[indices[0]]
                
                lb, ub = self._diffusion(neighbor_df.values.T)
                u_set = (neighbor_df.min(axis=0) + neighbor_df.max(axis=0)) / 2.0
                
                pool = self._get_pam_samples(lb, ub, u_set.values, pool_size=100)
                
                if len(pool) > 0:
                    distances = np.linalg.norm(pool - real_row_df.values, axis=1)
                    best_sample = pool[np.argmin(distances)]
                    surrogate_rows.append(best_sample)
                else:
                    surrogate_rows.append(real_row_df.values[0])

            if not surrogate_rows: continue
            
            surrogate_df = pd.DataFrame(surrogate_rows, columns=full_real_df.columns)
            
            if task_mode == 'classification':
                surrogate_y = surrogate_df[y.name].round().astype(int)
                n_classes = y.nunique()
                samples_per_class = int(np.ceil(obs_per_epoch / n_classes)) if n_classes > 0 else obs_per_epoch
                epoch_samples_list = []
                for class_label in y.unique():
                    class_surrogates = surrogate_df[surrogate_y == class_label]
                    if not class_surrogates.empty:
                        epoch_samples_list.append(
                            class_surrogates.sample(n=samples_per_class, replace=True, random_state=self.rng)
                        )
                if epoch_samples_list:
                    epoch_samples = pd.concat(epoch_samples_list)
                else:
                    epoch_samples = pd.DataFrame(columns=full_real_df.columns)
            else:
                epoch_samples = surrogate_df.sample(n=obs_per_epoch, replace=True, random_state=self.rng)
            
            all_synthetic_data.append(epoch_samples)
            yield pd.concat(all_synthetic_data, ignore_index=True)
