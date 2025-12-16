import torch
from torch import nn

class ResidualMLPBlock(nn.Module):
    def __init__(self, channels, expansion_factor=16):
        super(ResidualMLPBlock, self).__init__() 
        hidden_dim = int(channels*expansion_factor)
        self.fc1 = nn.Linear(channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, channels)
        self.activation = nn.GELU() 
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # Residual connection
        residual = x.clone() 
        x = self.norm(x)
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out + residual

class UW_Estimator(nn.Module):
    def __init__(self, input_length=3, hidden_channels=16):
        super(UW_Estimator, self).__init__() 
        
        self.hidden_channels = hidden_channels
        self.fc_in = nn.Linear(input_length, hidden_channels)
        self.res_block = ResidualMLPBlock(hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, 1) 
        nn.init.zeros_(self.fc_out.weight) 
        nn.init.ones_(self.fc_out.bias)

    def forward(self, x, task_list):

        x = self.fc_in(x)
        x = self.res_block(x)
        x = self.fc_out(x) 

        return x.reshape(-1)

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, task_list = []):
        super(AutomaticWeightedLoss, self).__init__() 
        self.weight_estimator = UW_Estimator() 
        self.loss_fun = nn.L1Loss(reduction='none') 

    def forward(self, in_pic, label_pic, restored_pic, task_list = []): 
        
        loss_input = self.loss_fun(in_pic, label_pic).view(in_pic.shape[0], -1).mean(dim=1) 
        loss_change = self.loss_fun(in_pic, restored_pic).view(in_pic.shape[0], -1).mean(dim=1) 
        loss_restored = self.loss_fun(restored_pic, label_pic).view(in_pic.shape[0], -1).mean(dim=1) 
        loss_combine = torch.stack([loss_input, loss_change, loss_restored], dim=1).clone().detach() 
        
        loss_sum = 0 
        uncertainty_weight = self.weight_estimator(loss_combine, task_list)
        for i, loss in enumerate(loss_restored):
            loss_sum += 0.5 / (uncertainty_weight[i] ** 2 + 1e-8) * loss + 0.5*torch.log(1 + uncertainty_weight[i] ** 2)
        return loss_sum/len(loss_restored) 
