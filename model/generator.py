import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # Camadas Hidden
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        
        # Camadas totalmente conectadas
        self.fc6 = nn.Linear(hidden_dim, output_size)
        
        # Camada de Dropout 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
      
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), 0.2)
        x = self.dropout(x)
        
        # Camada final com função de ativação tanh
        out = F.tanh(self.fc6(x))

        return out
