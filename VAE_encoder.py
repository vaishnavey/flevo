import torch
import torch.nn as nn

class VAE_MLP_encoder(nn.Module):
    """
    MLP encoder class for the VAE model using ESM embeddings.
    
    Expected input shape: (batch_size, seq_len, embedding_size)
    Example: ESM-2 outputs (N, 200, 1280) → set seq_len=200, input_size=1280
    """
    def __init__(self, params):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = params['seq_len']
        self.embedding_size = params['input_size']
        self.hidden_layers_sizes = params['hidden_layers_sizes']
        self.z_dim = params['z_dim']
        self.dropout_proba = params['dropout_proba']

        self.mu_bias_init = 0.1
        self.log_var_bias_init = -10.0

        if self.dropout_proba > 0.0:
            self.dropout_layer = nn.Dropout(p=self.dropout_proba)

        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'linear': nn.Identity()
        }
        self.nonlinear_activation = activation_map.get(params['nonlinear_activation'], nn.ReLU())

        input_dim = self.seq_len * self.embedding_size
        self.hidden_layers = nn.ModuleDict()
        for layer_index, hidden_size in enumerate(self.hidden_layers_sizes):
            in_dim = input_dim if layer_index == 0 else self.hidden_layers_sizes[layer_index - 1]
            layer = nn.Linear(in_dim, hidden_size)
            nn.init.constant_(layer.bias, self.mu_bias_init)
            self.hidden_layers[str(layer_index)] = layer

        self.fc_mean = nn.Linear(self.hidden_layers_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_mean.bias, self.mu_bias_init)

        self.fc_log_var = nn.Linear(self.hidden_layers_sizes[-1], self.z_dim)
        nn.init.constant_(self.fc_log_var.bias, self.log_var_bias_init)

    def forward(self, x):
        assert x.shape[1:] == (self.seq_len, self.embedding_size), \
            f"Expected input of shape (batch, {self.seq_len}, {self.embedding_size}), got {x.shape}"

        if self.dropout_proba > 0.0:
            x = self.dropout_layer(x)

        x = x.view(-1, self.seq_len * self.embedding_size)

        for layer_index in range(len(self.hidden_layers_sizes)):
            x = self.nonlinear_activation(self.hidden_layers[str(layer_index)](x))
            if self.dropout_proba > 0.0:
                x = self.dropout_layer(x)

        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var
