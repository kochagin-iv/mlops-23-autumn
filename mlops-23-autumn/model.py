import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 196)
        self.fc5 = nn.Linear(196, 32)
        self.fc6 = nn.Linear(32, 6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.bn(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x


def make_model(input_dim: int):
    model = MyModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # print(model)
    return model, criterion, optimizer
