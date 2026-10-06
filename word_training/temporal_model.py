"""Preserve the short sound sequence and following context of a whole word."""
import torch
from torch import nn
from wake_model import WhisperWakeModel


class TemporalTinyWakeModel(WhisperWakeModel):
    def __init__(self,config):
        super().__init__(config)
        self.temporal=nn.ModuleList([])
        for dilation in [1,2,4]:
            output=nn.Conv1d(96,96,1)
            nn.init.zeros_(output.weight);nn.init.zeros_(output.bias)
            self.temporal.append(nn.Sequential(nn.Conv1d(96,96,5,padding=2*dilation,dilation=dilation,groups=96),
                nn.Conv1d(96,96,1),nn.GroupNorm(8,96),nn.SiLU(),nn.Dropout(.1),output))
    def forward(self,value):
        value=self.encoder(value.squeeze(1),return_dict=False)[0].transpose(1,2)
        value=self.project(value)
        for block in self.temporal:value=value+block(value)
        attention=torch.softmax(self.attention(value),dim=-1)
        return self.head(torch.cat([(value*attention).sum(-1),value.amax(-1)],dim=1)).squeeze(-1)
