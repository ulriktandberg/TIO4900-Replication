import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
 
class _TwoTowerMLPNetwork(nn.Module):
    """
    Constructs a two-tower feedforward architecture: one for forward rates, one for macro/fred variables.
    Towers merge before the output layer. The macro tower includes dropout regularization.
    """
    def __init__(self, input_dim_fwd, input_dim_fred, archi_fwd, archi_fred, output_dim, dropout_rate=0.0, activation="tanh"):
        super(_TwoTowerMLPNetwork, self).__init__()
       
        # Forward rates tower
        fwd_layers = []
        current_dim = input_dim_fwd
        for i, hidden_dim in enumerate(archi_fwd):
            fwd_layers.append(nn.Linear(current_dim, hidden_dim))
            fwd_layers.append(_activation_layer(activation))
            if i == len(archi_fwd) - 1:
                fwd_layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.fwd_tower = nn.Sequential(*fwd_layers)
        self.fwd_tower_out_dim = current_dim
       
        # Macro/fred tower with dropout regularization
        fred_layers = []
        current_dim = input_dim_fred
        for i, hidden_dim in enumerate(archi_fred):
            fred_layers.append(nn.Linear(current_dim, hidden_dim))
            fred_layers.append(_activation_layer(activation))
            if dropout_rate > 0:
                fred_layers.append(nn.Dropout(dropout_rate))
            if i == len(archi_fred) - 1:
                fred_layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.fred_tower = nn.Sequential(*fred_layers)
        self.fred_tower_out_dim = current_dim
       
        # Merge batch norm and output layer
        merged_dim = self.fwd_tower_out_dim + self.fred_tower_out_dim
        self.merge_bn = nn.BatchNorm1d(merged_dim)
        self.output = nn.Linear(merged_dim, output_dim)
       
    def forward(self, x_fwd, x_fred):
        h_fwd = self.fwd_tower(x_fwd)
        h_fred = self.fred_tower(x_fred)
        h_merged = torch.cat([h_fwd, h_fred], dim=1)
        h_merged = self.merge_bn(h_merged)
        return self.output(h_merged)
 
class MacroForwardANNWrapper:
    """
    A scikit-learn style wrapper for the PyTorch two-tower MLP.
    Accepts X with MultiIndex ('forward', 'fred') and trains separate towers with separate scalers.
    """
    def __init__(self, archi_forward=(3,), archi_macro=(16, 8), lr=0.01, epochs=100, warm_start=False,
                 seed=42, momentum=0.9, param_grid=None, tune_every=60, patience=10,
                 y_center=True, activation="relu"):
        self.archi_forward = archi_forward
        self.archi_macro = archi_macro
        self.lr = lr
        self.epochs = epochs
        self.warm_start = warm_start
        self.random_state = seed
        self.momentum = momentum
        self.param_grid = param_grid if param_grid is not None else {
            'penalty': [0.001, 0.0001],
            'dropout_rate': [0.0, 0.1, 0.2]
        }
        self.tune_every = tune_every
        self.patience = patience
        self.y_center = y_center
        self.activation = activation
       
        # Internal state
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.val_criterion = nn.MSELoss(reduction='none')
        self.best_params_ = None
        self._fit_calls = 0
        self.val_loss_ = None
       
        # Scalers for each input block
        self.x_scaler_forward = None
        self.x_scaler_fred = None
        self.y_scaler = None
 
    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
           
    def _extract_blocks(self, X):
        """
        Extract forward and fred blocks from MultiIndex DataFrame.
        Returns numpy arrays for each block.
        """
        if isinstance(X, pd.DataFrame):
            if 'forward' in X.columns.get_level_values(0):
                X_fwd = X['forward'].values
            else:
                raise ValueError("'forward' not found in X columns")
           
            if 'fred' in X.columns.get_level_values(0):
                X_fred = X['fred'].values
            else:
                raise ValueError("'fred' not found in X columns")
        else:
            raise ValueError("X must be a pandas DataFrame with MultiIndex columns")
       
        return X_fwd, X_fred
   
    def _extract_y(self, y):
        """Extract y as a numpy array."""
        if hasattr(y, 'values'):
            arr = y.values
        else:
            arr = np.array(y)
       
        if arr.ndim == 1:
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
        Fits the two-tower neural network.
        """
        X_fwd_arr, X_fred_arr = self._extract_blocks(X)
        y_arr = self._extract_y(y)
       
        # Always refit scalers on the current expanding window's training set
        if not self.warm_start or self.x_scaler_forward is None:
            self.x_scaler_forward = StandardScaler()
            self.x_scaler_fred = StandardScaler()
            self.y_scaler = StandardScaler(with_mean=self.y_center, with_std=True)
           
        X_fwd_scaled = self.x_scaler_forward.fit_transform(X_fwd_arr)
        X_fred_scaled = self.x_scaler_fred.fit_transform(X_fred_arr)
        y_scaled = self.y_scaler.fit_transform(y_arr)
       
        X_fwd_tensor = torch.tensor(X_fwd_scaled, dtype=torch.float32)
        X_fred_tensor = torch.tensor(X_fred_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
 
        input_dim_fwd = X_fwd_tensor.shape[1]
        input_dim_fred = X_fred_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        n_samples = X_fwd_tensor.shape[0]
 
        split = int(n_samples * 0.85)
 
        # 2. Hyperparameter tuning loop
        if self._should_tune() and split >= 10 and (n_samples - split) >= 3:
            X_fwd_subtrain, X_fwd_val = X_fwd_tensor[:split], X_fwd_tensor[split:]
            X_fred_subtrain, X_fred_val = X_fred_tensor[:split], X_fred_tensor[split:]
            y_subtrain, y_val = y_tensor[:split], y_tensor[split:]
           
            best_mse = float('inf')
            best_penalty = self.param_grid['penalty'][0]
            best_dropout_rate = self.param_grid.get('dropout_rate', [0.0])[0]
            best_epochs = self.epochs
           
            for penalty in self.param_grid['penalty']:
                for dropout_rate in self.param_grid.get('dropout_rate', [0.0]):
                    self._set_seed()
                    temp_model = _TwoTowerMLPNetwork(
                        input_dim_fwd=input_dim_fwd,
                        input_dim_fred=input_dim_fred,
                        archi_fwd=self.archi_forward,
                        archi_fred=self.archi_macro,
                        output_dim=output_dim,
                        dropout_rate=dropout_rate,
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
                        preds = temp_model(X_fwd_subtrain, X_fred_subtrain)
                        loss = self.criterion(preds, y_subtrain)
                        if penalty > 0:
                            l1_penalty = sum(p.abs().sum() for p in temp_model.parameters())
                            loss += penalty * l1_penalty
                        loss.backward()
                        temp_optimizer.step()
                   
                        # Early Stopping Check against validation set every epoch
                        temp_model.eval()
                        with torch.no_grad():
                            val_preds = temp_model(X_fwd_val, X_fred_val)
                            val_mse = self.criterion(val_preds, y_val).item()
                           
                        early_stopper(val_mse, epoch)
                        if early_stopper.early_stop:
                            break
                       
                    if early_stopper.best_loss < best_mse:
                        best_mse = early_stopper.best_loss
                        best_penalty = penalty
                        best_dropout_rate = dropout_rate
                        best_epochs = early_stopper.best_epoch + 1
                   
            self.best_params_ = {'penalty': best_penalty, 'dropout_rate': best_dropout_rate, 'epochs': best_epochs}
        elif self.best_params_ is None:
            self.best_params_ = {
                'penalty': self.param_grid['penalty'][0],
                'dropout_rate': self.param_grid.get('dropout_rate', [0.0])[0],
                'epochs': self.epochs
            }
 
        current_penalty = self.best_params_['penalty']
        current_dropout_rate = self.best_params_['dropout_rate']
        current_epochs = self.best_params_['epochs']
       
        # 3. Check if we need to initialize or re-initialize the model
        if self.model is None or not self.warm_start:
            self._set_seed()
            self.model = _TwoTowerMLPNetwork(
                input_dim_fwd=input_dim_fwd,
                input_dim_fred=input_dim_fred,
                archi_fwd=self.archi_forward,
                archi_fred=self.archi_macro,
                output_dim=output_dim,
                dropout_rate=current_dropout_rate,
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
            predictions = self.model(X_fwd_tensor, X_fred_tensor)
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
                val_preds = self.model(X_fwd_tensor[split:], X_fred_tensor[split:])
                per_output_losses = self.val_criterion(val_preds, y_tensor[split:]).mean(dim=0)
                self.val_loss_ = per_output_losses.numpy()
        else:
            self.val_loss_ = np.full(output_dim, np.nan)
           
        self._fit_calls += 1
        return self
 
    def predict(self, X):
        if self.model is None:
            raise ValueError("This model instance is not fitted yet. Call 'fit' before 'predict'.")
           
        X_fwd_arr, X_fred_arr = self._extract_blocks(X)
        X_fwd_scaled = self.x_scaler_forward.transform(X_fwd_arr)
        X_fred_scaled = self.x_scaler_fred.transform(X_fred_arr)
        X_fwd_tensor = torch.tensor(X_fwd_scaled, dtype=torch.float32)
        X_fred_tensor = torch.tensor(X_fred_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X_fwd_tensor, X_fred_tensor).numpy()
           
        # Inverse transform the predictions to return back to raw scale
        preds = self.y_scaler.inverse_transform(preds_scaled)
           
        if preds.shape[1] == 1:
            return preds.flatten()
        return preds
