import torch
import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self,temperature=3.4608009467937):
        super().__init__()
        self.temp = temperature

        self.lossfn = nn.KLDivLoss(reduction='sum',log_target=True)

    def forward(self,model_preds_list,targets_list):
        total_loss = 0.0
        num_race = len(model_preds_list)

        for model_preds,target in zip(model_preds_list,targets_list):
            model_preds = model_preds.squeeze()
            pred_log_prob = F.log_softmax(model_preds,dim = 0)

            target_log_prob = F.log_softmax(target/self.temp,dim=0)

            race_loss = self.lossfn(pred_log_prob,target_log_prob)
            total_loss += race_loss
        
        return total_loss/num_race

