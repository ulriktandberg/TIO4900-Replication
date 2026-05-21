import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _activation_layer(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be either 'relu' or 'tanh'")
 
class EarlyStopping:
    """
    Simple and reusable early stopping module to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0
 
    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
 
class _MLPNetwork(nn.Module):
    """
    Constructs a simple feedforward architecture based on the `archi` tuple using forward rates only.
    """
    def __init__(self, input_dim, archi, output_dim, activation="relu"):
        super(_MLPNetwork, self).__init__()
       
        layers = []
        current_dim = input_dim
       
        # Build hidden layers
        for i, hidden_dim in enumerate(archi):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_activation_layer(activation))

            # Apply Batch Normalization to the activations after the last ReLU layer
            if i == len(archi) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
               
            current_dim = hidden_dim
           
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
       
        # Pack layers into a sequential block
        self.network = nn.Sequential(*layers)
       
    def forward(self, x):
        return self.network(x)
 
class PyTorchMLPWrapper:
    """
    A scikit-learn style wrapper for the PyTorch MLP.
    Constructs a simple feedforward architecture based on the `archi` tuple using forward rates only.
    """
    def __init__(self, archi=(3,), lr=0.01, epochs=100, warm_start=False,
                 seed=42, momentum=0.9, param_grid=None, tune_every=60, patience=10,
                 use_pca=False, n_components=None, y_center=True, activation="relu"):
        self.archi = archi
        self.lr = lr
        self.epochs = epochs
        self.warm_start = warm_start
        self.random_state = seed
        self.momentum = momentum
        self.param_grid = param_grid if param_grid is not None else {'penalty': [0.001, 0.0001]}
        self.tune_every = tune_every
        self.patience = patience
        self.use_pca = use_pca
        self.n_components = n_components
        self.y_center = y_center
        self.activation = activation
       
        # Internal state
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss() # Standard MSE for regression
        self.val_criterion = nn.MSELoss(reduction='none') # For per-output validation loss
        self.best_params_ = None
        self._fit_calls = 0
        self.val_loss_ = None # Will store per-output array
       
        # Scalers
        self.x_scaler = None
        self.y_scaler = None
        self.pca = None

    def _transform_features(self, X_arr, fit=False):
        X_scaled = self.x_scaler.fit_transform(X_arr) if fit else self.x_scaler.transform(X_arr)

        if not self.use_pca:
            return X_scaled

        if fit:
            n_features = X_scaled.shape[1]
            if self.n_components is None:
                n_components = n_features
            else:
                n_components = min(self.n_components, n_features)
            self.pca = PCA(n_components=n_components)
            return self.pca.fit_transform(X_scaled)

        return self.pca.transform(X_scaled)
 
    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
           
    def _extract_array(self, data, is_X=True):
        """Helper to extract pure numpy arrays from potentially complex input structures."""
        if is_X and isinstance(data, pd.DataFrame) and 'forward' in data:
            data = data['forward']
           
        if hasattr(data, 'values'):
            arr = data.values
        else:
            arr = np.array(data)
           
        if not is_X and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
           
        return arr
 
    def _should_tune(self):
        if self.best_params_ is None:
            return True
        if self.tune_every is None or self.tune_every <= 1:
            return True
        return (self._fit_calls % self.tune_every) == 0
 
    def fit(self, X, y):
        """
        Fits the neural network.
        """
        X_arr = self._extract_array(X, is_X=True)
        y_arr = self._extract_array(y, is_X=False)
       
        # Always refit scalers on the current expanding window's training set
        if not self.warm_start or self.x_scaler is None:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler(with_mean=self.y_center, with_std=True)

        X_scaled = self._transform_features(X_arr, fit=True)
        y_scaled = self.y_scaler.fit_transform(y_arr)
       
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
 
        input_dim = X_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        n_samples = X_tensor.shape[0]
 
        split = int(n_samples * 0.85)
 
        # 2. Hyperparameter tuning loop
        if self._should_tune() and split >= 10 and (n_samples - split) >= 3:
            X_subtrain, X_val = X_tensor[:split], X_tensor[split:]
            y_subtrain, y_val = y_tensor[:split], y_tensor[split:]
           
            best_mse = float('inf')
            best_penalty = self.param_grid['penalty'][0]
            best_epochs = self.epochs
           
            for penalty in self.param_grid['penalty']:
                self._set_seed()
                temp_model = _MLPNetwork(
                    input_dim=input_dim,
                    archi=self.archi,
                    output_dim=output_dim,
                    activation=self.activation,
                )
                # temp_optimizer = optim.SGD(
                #     temp_model.parameters(), lr=self.lr, momentum=self.momentum,
                #     nesterov=True, weight_decay=penalty
                # )
                
                temp_optimizer = optim.Adam(
                    temp_model.parameters(),
                    lr=self.lr,
                    weight_decay=penalty,
                )
               
                early_stopper = EarlyStopping(patience=self.patience)
               
                for epoch in range(self.epochs):
                    temp_model.train()
                    temp_optimizer.zero_grad()
                    preds = temp_model(X_subtrain)
                    loss = self.criterion(preds, y_subtrain)
                    if penalty > 0:
                        l1_penalty = sum(p.abs().sum() for p in temp_model.parameters())
                        loss += penalty * l1_penalty
                    loss.backward()
                    temp_optimizer.step()
               
                    # Early Stopping Check against validation set every epoch
                    temp_model.eval()
                    with torch.no_grad():
                        val_preds = temp_model(X_val)
                        val_mse = self.criterion(val_preds, y_val).item()
                       
                    early_stopper(val_mse, epoch)
                    if early_stopper.early_stop:
                        break
                   
                if early_stopper.best_loss < best_mse:
                    best_mse = early_stopper.best_loss
                    best_penalty = penalty
                    best_epochs = early_stopper.best_epoch + 1 # +1 since 0-indexed
                   
            self.best_params_ = {'penalty': best_penalty, 'epochs': best_epochs}
        elif self.best_params_ is None:
            self.best_params_ = {'penalty': self.param_grid['penalty'][0], 'epochs': self.epochs}
 
        current_penalty = self.best_params_['penalty']
        current_epochs = self.best_params_['epochs']
 
        # 3. Check if we need to initialize or re-initialize the model
        if self.model is None or not self.warm_start:
            self._set_seed()
            self.model = _MLPNetwork(
                input_dim=input_dim,
                archi=self.archi,
                output_dim=output_dim,
                activation=self.activation,
            )
           
            # self.optimizer = optim.SGD(
            #     self.model.parameters(),
            #     lr=self.lr,
            #     momentum=self.momentum,
            #     nesterov=True,
            #     weight_decay=current_penalty
            # )

            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=current_penalty,
            )
        else:
            # If using warm start, ensure the optimizer uses the current best penalty for L2 weight decay
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = current_penalty
 
        # 4. Training Loop on full dataset
        self.model.train()
        for epoch in range(current_epochs):
            self.optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = self.criterion(predictions, y_tensor)
           
            # Application of L1 penalty using the tuned parameter
            if current_penalty > 0:
                l1_penalty = sum(p.abs().sum() for p in self.model.parameters())
                loss += current_penalty * l1_penalty
           
            loss.backward()
            self.optimizer.step()
            
        # --- Evaluate Validation Loss for Ensembling ---
        if split >= 10 and (n_samples - split) > 0:
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_tensor[split:])
                # Calculate mean squared error across the batch dimension (dim 0), 
                # leaving us with an array of losses for each output (dim 1)
                per_output_losses = self.val_criterion(val_preds, y_tensor[split:]).mean(dim=0)
                self.val_loss_ = per_output_losses.numpy()
        else:
            self.val_loss_ = np.full(output_dim, np.nan)
           
        self._fit_calls += 1
        return self
 
    def predict(self, X):
        if self.model is None:
            raise ValueError("This model instance is not fitted yet. Call 'fit' before 'predict'.")
           
        X_arr = self._extract_array(X, is_X=True)
        X_scaled = self._transform_features(X_arr, fit=False)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).numpy()
           
        # Inverse transform the predictions to return back to raw scale
        preds = self.y_scaler.inverse_transform(preds_scaled)
           
        if preds.shape[1] == 1:
            return preds.flatten()
        return preds