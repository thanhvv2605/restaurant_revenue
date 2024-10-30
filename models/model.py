import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(SimpleNN, self).__init__()
        layers = []
        in_dim = input_dim
        # Tạo các lớp ẩn
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            in_dim = units
        # Thêm lớp đầu ra
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)