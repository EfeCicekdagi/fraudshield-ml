import numpy as np
import json
import os

class StableIsotonicCalibrator:
    def __init__(self, metadata: dict):
        self.x_thresholds = np.array(metadata.get('x_thresholds', []))
        self.y_thresholds = np.array(metadata.get('y_thresholds', []))
        self.X_min = metadata.get('X_min', 0.0)
        self.X_max = metadata.get('X_max', 1.0)
        self.out_of_bounds = metadata.get('out_of_bounds', 'clip')
        
        # Scikit-learn Isotonic interpolator stores step functions via x_thresholds and y_thresholds.
        # np.interp with these thresholds linearly interpolates between them, matching scikit-learn
        # perfectly for intermediate points because the scikit-learn model outputs piece-wise constant
        # values and at the thresholds it corresponds exactly to the y values.
        # Wait, scikit-learn isotonic interpolation is actually piece-wise constant. 
        # But wait! If `interpolation` is 'linear' (not 'constant'), scikit-learn interpolates!
        # And the default interpolation is 'linear'.
        
    def predict(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)
        
        if self.out_of_bounds == 'clip':
            T = np.clip(T, self.X_min, self.X_max)
            
        res = np.interp(T, self.x_thresholds, self.y_thresholds)
        return res

class StablePreprocessor:
    def __init__(self, metadata: dict, arrays: dict):
        self.transformers = metadata.get('transformers', [])
        self.arrays = arrays
        self.expected_columns = []
        
        # Reconstruct expected columns in order
        for t in self.transformers:
            self.expected_columns.extend(t['columns'])
            
    def transform(self, df) -> np.ndarray:
        # Pre-allocate output or collect chunks
        chunks = []
        
        for t in self.transformers:
            cols = t['columns']
            X_chunk = df[cols].values
            
            if t['type'] == 'StandardScaler':
                mean = self.arrays[t['arrays'][0]]
                scale = self.arrays[t['arrays'][1]]
                # Safe division just like sklearn
                scale = np.where(scale == 0, 1.0, scale)
                X_chunk = (X_chunk - mean) / scale
                chunks.append(X_chunk)
                
            elif t['type'] == 'OneHotEncoder':
                # categories arrays
                arr_names = t['arrays']
                categories_list = [self.arrays[name] for name in arr_names]
                handle_unknown = t.get('handle_unknown', 'ignore')
                
                # Transform each column
                ohe_cols = []
                for i in range(X_chunk.shape[1]):
                    col_data = X_chunk[:, i]
                    cats = categories_list[i]
                    
                    # Create one-hot matrix for this column
                    one_hot = np.zeros((len(col_data), len(cats)), dtype=np.float64)
                    
                    # Cast the column data to match the category array dtype type if needed
                    # Especially important since we casted object arrays to string in the bundle
                    is_string_cat = cats.dtype.kind in {'U', 'S'}
                    
                    for j, val in enumerate(col_data):
                        if is_string_cat:
                            val = str(val)
                            
                        # Find index
                        idx = np.where(cats == val)[0]
                        if len(idx) > 0:
                            one_hot[j, idx[0]] = 1.0
                        else:
                            if handle_unknown == 'error':
                                raise ValueError(f"Found unknown category {val} in column {i}")
                            # ignore means all zeros, which is already the default for one_hot
                    ohe_cols.append(one_hot)
                    
                if ohe_cols:
                    chunks.append(np.hstack(ohe_cols))
                    
        return np.hstack(chunks)

    def get_feature_names_out(self):
        names = []
        for t in self.transformers:
            cols = t['columns']
            if t['type'] == 'StandardScaler':
                names.extend([f"num__{c}" for c in cols])
            elif t['type'] == 'OneHotEncoder':
                arr_names = t['arrays']
                categories_list = [self.arrays[name] for name in arr_names]
                for c, cats in zip(cols, categories_list):
                    names.extend([f"cat__{c}_{cat}" for cat in cats])
        return names
