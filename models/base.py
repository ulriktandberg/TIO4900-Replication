import sklearn
import numpy as np

class LinearModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class RandomWalkModel:
    """Diagnostic only — predicts y[t] = y[t-1].
    
    With overlapping returns, this achieves high apparent R² due to
    mechanical autocorrelation (~11/12 months shared) and lookahead,
    NOT genuine predictive power. Use to check whether other models
    are simply mimicking this persistence.
    """
    def __init__(self):
        self.y_last = None

    def fit(self, X, y):
        # Store the last observed y value
        self.y_last = y[-1]
    
    def predict(self, X):
        # For Random Walk, always predict the last observed value
        n_pred = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.full(n_pred, self.y_last)
    
    
class HistoricalMeanModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean = np.mean(y)
    
    def predict(self, X):
        return np.array(self.mean)
    


class PCABaselineModel:
    def __init__(self, components=3, series='yields'):
        self.components = components
        self.series = series
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        # perform PCA on yields:
        yields = X[self.series]
        # Fit the PCA on the TRAINING set
        pca_scores = self.pca.fit_transform(yields)
        
        self.model.fit(pca_scores, y)
    
    def predict(self, X):
        yields = X[self.series]
        pca_scores = self.pca.transform(yields)
        return self.model.predict(pca_scores)


class MacroPCA8Model:
    """
    Panel A, model 1:
    PCR using the first 8 PCs of the macro block.
    """

    def __init__(self, components=8, series='fred'):
        self.components = components
        self.series = series

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        macro = X[self.series]
        macro_scaled = self.scaler.fit_transform(macro)
        macro_pcs = self.pca.fit_transform(macro_scaled)
        self.model.fit(macro_pcs, y)

    def predict(self, X):
        macro = X[self.series]
        macro_scaled = self.scaler.transform(macro)
        macro_pcs = self.pca.transform(macro_scaled)
        return self.model.predict(macro_pcs)

    
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing

class PCABaselineModelPlusN:
    def __init__(self, components=3, series='yields', n_extra=1):
        self.components = components
        self.series = series
        self.n_extra = n_extra
        
        # Add Scalers for both data sources
        self.scaler_series = sklearn.preprocessing.StandardScaler()
        self.scaler_fred = sklearn.preprocessing.StandardScaler()
        
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.fred_pca = sklearn.decomposition.PCA(n_components=self.n_extra)
        
        # Scale the final concatenated features so the linear model coefficients are comparable
        self.scaler_features = sklearn.preprocessing.StandardScaler()
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        # 1. Scale and PCA on yields/forwards
        yields = X[self.series]
        yields_scaled = self.scaler_series.fit_transform(yields)
        pca_scores = self.pca.fit_transform(yields_scaled)

        # 2. Scale and PCA on 'fred'
        fred = X['fred']
        fred_scaled = self.scaler_fred.fit_transform(fred)
        fred_pc1 = self.fred_pca.fit_transform(fred_scaled)  # shape (n_samples, n_extra)

        # 3. Concatenate and scale final features
        features = np.concatenate([pca_scores, fred_pc1], axis=1)
        features_scaled = self.scaler_features.fit_transform(features)

        self.model.fit(features_scaled, y)
    
    def predict(self, X):
        # 1. Transform yields
        yields = X[self.series]
        yields_scaled = self.scaler_series.transform(yields)
        pca_scores = self.pca.transform(yields_scaled)

        # 2. Transform fred
        fred = X['fred']
        fred_scaled = self.scaler_fred.transform(fred)
        fred_pc1 = self.fred_pca.transform(fred_scaled)

        # 3. Concatenate, transform features, and predict
        features = np.concatenate([pca_scores, fred_pc1], axis=1)
        features_scaled = self.scaler_features.transform(features)
        
        return self.model.predict(features_scaled)
    

class PCABaselineModelMacroGroups:
    def __init__(self, components=3, series='yields', lasso=False, alpha=0.01, macro_pcs=1):
        self.components = components
        self.series = series
        self.lasso = lasso
        self.alpha = alpha
        self.macro_pcs = macro_pcs
        
        # Scalers for inputs
        self.scaler_series = sklearn.preprocessing.StandardScaler()
        self.scaler_fred = sklearn.preprocessing.StandardScaler()
        
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.fred_pcas = {}  # Will hold one PCA per macro category
        
        self.scaler_features = sklearn.preprocessing.StandardScaler()
        
        if self.lasso:
            self.model = sklearn.linear_model.Lasso(alpha=self.alpha)
        else:
            self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y):
        # 1. Scale and PCA on yields/forwards
        yields = X[self.series]
        yields_scaled = self.scaler_series.fit_transform(yields)
        pca_scores = self.pca.fit_transform(yields_scaled)

        # 2. Scale 'fred' and rebuild DataFrame to preserve the MultiIndex
        fred = X['fred']
        fred_scaled_arr = self.scaler_fred.fit_transform(fred)
        fred_scaled = pd.DataFrame(fred_scaled_arr, index=fred.index, columns=fred.columns)

        # 3. PCA on each macro category using the scaled DataFrame
        macro_cat_pc1s = []
        self.fred_pcas = {}
        for cat in fred_scaled.columns.get_level_values(0).unique():
            cat_df = fred_scaled[cat]
            pca = sklearn.decomposition.PCA(n_components=self.macro_pcs)
            pc1 = pca.fit_transform(cat_df)
            macro_cat_pc1s.append(pc1)
            self.fred_pcas[cat] = pca

        macro_cat_pc1s = np.hstack(macro_cat_pc1s)  # shape (n_samples, n_cats)

        # 4. Concatenate and scale final features
        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        features_scaled = self.scaler_features.fit_transform(features)
        
        self.model.fit(features_scaled, y)

    def predict(self, X):
        # 1. Transform yields
        yields = X[self.series]
        yields_scaled = self.scaler_series.transform(yields)
        pca_scores = self.pca.transform(yields_scaled)

        # 2. Transform fred and rebuild MultiIndex DataFrame
        fred = X['fred']
        fred_scaled_arr = self.scaler_fred.transform(fred)
        fred_scaled = pd.DataFrame(fred_scaled_arr, index=fred.index, columns=fred.columns)

        # 3. Transform macro categories
        macro_cat_pc1s = []
        for cat, pca in self.fred_pcas.items():
            cat_df = fred_scaled[cat]
            pc1 = pca.transform(cat_df)
            macro_cat_pc1s.append(pc1)
        macro_cat_pc1s = np.hstack(macro_cat_pc1s)

        # 4. Concatenate, transform final features, and predict
        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        features_scaled = self.scaler_features.transform(features)
        
        return self.model.predict(features_scaled)
    
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.linear_model

class PCABaselineModelMacroGroupsHYPERGRID:
    def __init__(self, components=3, series='forward', lasso=False, 
                 alphas=None, macro_pcs_list=None):
        self.components = components
        self.series = series
        self.lasso = lasso
        
        # Hyperparameter grids
        self.alphas = alphas if alphas is not None else[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
        self.macro_pcs_list = macro_pcs_list if macro_pcs_list is not None else [1]
        
        # Track best parameters found during fit
        self.best_alpha_ = None
        self.best_macro_pcs_ = None
        
        # Placeholders for fitted objects (instantiated dynamically during fit)
        self.scaler_series = None
        self.scaler_fred = None
        self.pca = None
        self.fred_pcas = {}
        self.scaler_features = None
        self.model = None

    def _fit_core(self, X, y, alpha, macro_pcs):
        """
        Internal function to fit scalers, PCAs, and the regression model
        on a specific dataset with a specific set of hyperparameters.
        """
        # 1. Initialize fresh scalers and PCAs to prevent state leakage
        self.scaler_series = sklearn.preprocessing.StandardScaler()
        self.scaler_fred = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=self.components)
        self.scaler_features = sklearn.preprocessing.StandardScaler()
        
        if self.lasso and alpha is not None:
            self.model = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
        else:
            self.model = sklearn.linear_model.LinearRegression()

        # 2. Scale and PCA on yields/forwards
        yields = X[self.series]
        yields_scaled = self.scaler_series.fit_transform(yields)
        pca_scores = self.pca.fit_transform(yields_scaled)

        # 3. Scale 'fred' and rebuild DataFrame to preserve the MultiIndex
        fred = X['fred']
        fred_scaled_arr = self.scaler_fred.fit_transform(fred)
        fred_scaled = pd.DataFrame(fred_scaled_arr, index=fred.index, columns=fred.columns)

        # 4. PCA on each macro category using the scaled DataFrame
        macro_cat_pc1s =[]
        self.fred_pcas = {}
        for cat in fred_scaled.columns.get_level_values(0).unique():
            cat_df = fred_scaled[cat]
            pca = sklearn.decomposition.PCA(n_components=macro_pcs)
            pc_scores = pca.fit_transform(cat_df)
            macro_cat_pc1s.append(pc_scores)
            self.fred_pcas[cat] = pca

        macro_cat_pc1s = np.hstack(macro_cat_pc1s)  # shape (n_samples, n_cats * macro_pcs)

        # 5. Concatenate and scale final features (CRITICAL for Lasso)
        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        features_scaled = self.scaler_features.fit_transform(features)
        
        self.model.fit(features_scaled, y)

    def fit(self, X, y):
        n_samples = len(y)
        split_idx = int(n_samples * 0.85)
        
        # Safely split X (handles both pandas DataFrame with MultiIndex or dict of DataFrames)
        if hasattr(X, 'iloc'):
            X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
        elif isinstance(X, dict):
            X_tr = {k: v.iloc[:split_idx] for k, v in X.items()}
            X_va = {k: v.iloc[split_idx:] for k, v in X.items()}
        else:
            X_tr, X_va = X[:split_idx], X[split_idx:]
            
        # Safely split y
        if hasattr(y, 'iloc'):
            y_tr, y_va = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            y_tr, y_va = y[:split_idx], y[split_idx:]

        best_loss = np.inf
        best_alpha = None
        best_macro_pcs = None

        alphas_to_test = self.alphas if self.lasso else [None]
        
        # Grid Search over 85/15 Split
        for alpha in alphas_to_test:
            for macro_pcs in self.macro_pcs_list:
                
                # Fit internal pipeline on 85% Train
                self._fit_core(X_tr, y_tr, alpha, macro_pcs)
                
                # Predict on 15% Validation
                preds = self.predict(X_va)
                
                # Calculate Validation MSE
                loss = np.mean((preds - np.array(y_va)) ** 2)
                
                # Track best hyperparameters
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
                    best_macro_pcs = macro_pcs

        self.best_alpha_ = best_alpha
        self.best_macro_pcs_ = best_macro_pcs
        
        # CRITICAL: Refit the entire pipeline on 100% of the expanding window data
        self._fit_core(X, y, best_alpha, best_macro_pcs)

    def predict(self, X):
        # 1. Transform yields
        yields = X[self.series]
        yields_scaled = self.scaler_series.transform(yields)
        pca_scores = self.pca.transform(yields_scaled)

        # 2. Transform fred and rebuild MultiIndex DataFrame
        fred = X['fred']
        fred_scaled_arr = self.scaler_fred.transform(fred)
        fred_scaled = pd.DataFrame(fred_scaled_arr, index=fred.index, columns=fred.columns)

        # 3. Transform macro categories
        macro_cat_pc1s =[]
        for cat, pca in self.fred_pcas.items():
            cat_df = fred_scaled[cat]
            pc_scores = pca.transform(cat_df)
            macro_cat_pc1s.append(pc_scores)
            
        macro_cat_pc1s = np.hstack(macro_cat_pc1s)

        # 4. Concatenate, transform final features, and predict
        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        features_scaled = self.scaler_features.transform(features)
        
        return self.model.predict(features_scaled)




class PCAFirst8PCsWithCPModel:
    def __init__(self, xr_full, cp_cols, components=8, n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.components = components
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=components)

        self.cp_model = sklearn.linear_model.LinearRegression()
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        fred = X['fred']
        forward = X['forward'].iloc[:, :self.n_cp_forwards]

        fred_scaled = self.scaler_macro.fit_transform(fred)
        fred_pcs = self.pca.fit_transform(fred_scaled)

        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        features = np.concatenate([cp_factor, fred_pcs], axis=1)
        self.model.fit(features, y)

    def predict(self, X):
        fred = X['fred']
        forward = X['forward'].iloc[:, :self.n_cp_forwards]

        fred_scaled = self.scaler_macro.transform(fred)
        fred_pcs = self.pca.transform(fred_scaled)

        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        features = np.concatenate([cp_factor, fred_pcs], axis=1)
        return self.model.predict(features)


class LudvigsonNgWithCPModel:
    def __init__(self, xr_full, cp_cols, n_factors=8, n_cp_forwards=5):
        self.xr_full = xr_full
        self.cp_cols = cp_cols
        self.n_factors = max(n_factors, 8)
        self.n_cp_forwards = n_cp_forwards

        self.scaler_macro = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=self.n_factors)

        self.cp_model = sklearn.linear_model.LinearRegression()
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        fred = X['fred']
        forward = X['forward'].iloc[:, :self.n_cp_forwards]

        fred_scaled = self.scaler_macro.fit_transform(fred)
        factors = self.pca.fit_transform(fred_scaled)

        cp_target = self.xr_full.loc[X.index, self.cp_cols].mean(axis=1)
        self.cp_model.fit(forward, cp_target)
        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]

        # features = np.concatenate([cp_factor, factors[:, :8]], axis=1)
        features = np.concatenate([cp_factor, F1, F1**3, F3, F4, F8], axis=1)
        self.model.fit(features, y)

    def predict(self, X):
        fred = X['fred']
        forward = X['forward'].iloc[:, :self.n_cp_forwards]

        fred_scaled = self.scaler_macro.transform(fred)
        factors = self.pca.transform(fred_scaled)

        cp_factor = self.cp_model.predict(forward).reshape(-1, 1)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]

        features = np.concatenate([cp_factor, F1, F1**3, F3, F4, F8], axis=1)
        # features = np.concatenate([cp_factor, factors[:, :8]], axis=1)
        return self.model.predict(features)



import numpy as np
import sklearn.decomposition
import sklearn.linear_model
import sklearn.preprocessing


class MacroPCA8Model:
    """
    Panel A, model 1:
    PCR using the first 8 PCs of the macro block.
    """

    def __init__(self, components=8, series='fred'):
        self.components = components
        self.series = series

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        macro = X[self.series]
        macro_scaled = self.scaler.fit_transform(macro)
        macro_pcs = self.pca.fit_transform(macro_scaled)
        self.model.fit(macro_pcs, y)

    def predict(self, X):
        macro = X[self.series]
        macro_scaled = self.scaler.transform(macro)
        macro_pcs = self.pca.transform(macro_scaled)
        return self.model.predict(macro_pcs)


class LudvigsonNgModelNew:
    """
    Panel A, model 2:
    Ludvigson-Ng specification:
        F1, F1^3, F3, F4, F8
    extracted from the first 8 macro PCs.
    """

    def __init__(self, series='fred', n_factors=8):
        self.series = series
        self.n_factors = max(n_factors, 8)

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.pca = sklearn.decomposition.PCA(n_components=self.n_factors)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        macro = X[self.series]
        macro_scaled = self.scaler.fit_transform(macro)
        factors = self.pca.fit_transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]

        features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)
        self.model.fit(features, y)

    def predict(self, X):
        macro = X[self.series]
        macro_scaled = self.scaler.transform(macro)
        factors = self.pca.transform(macro_scaled)

        F1 = factors[:, [0]]
        F3 = factors[:, [2]]
        F4 = factors[:, [3]]
        F8 = factors[:, [7]]

        features = np.concatenate([F1, F1**3, F3, F4, F8], axis=1)
        return self.model.predict(features)