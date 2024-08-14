import torch.nn as nn


class MLP(nn.Module):
    """Basic multi-layer perceptron (MLP) model."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        # Create the input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Create the hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Create the output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x
