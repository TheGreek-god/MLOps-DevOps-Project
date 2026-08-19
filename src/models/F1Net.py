import torch
import torch.nn as nn
import torch.nn.functional as F



class F1Net(nn.Module):
    def __init__(self,num_teams,num_drivers,embedding_dim=16):
        super().__init__()

        self.team_embedding = nn.Embedding(num_teams+10,embedding_dim)
        self.driver_embedding = nn.Embedding(num_drivers*4,embedding_dim)

        self.fc1 = nn.Linear(5+(2*embedding_dim),128)
        self.fc2 = nn.Linear(128,128)
        self.output = nn.Linear(128,1)

        self.dropout = nn.Dropout(p=.25)



    def forward(self,x,team_idx,driver_idx):
        team_vec = self.team_embedding(team_idx)
        driver_vec = self.driver_embedding(driver_idx)

        x = torch.concat([x,team_vec,driver_vec],dim=1)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))

        x = self.output(x)

        return x
