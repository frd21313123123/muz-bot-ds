"""Compact acoustic CTC student; six target/filler states, no ASR vocabulary."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,source,target,stride):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(source,target,3,padding=1,stride=stride,bias=False),
                                nn.GroupNorm(8,target),nn.SiLU(),nn.Conv2d(target,target,3,padding=1,bias=False),nn.GroupNorm(8,target))
        self.skip=nn.Conv2d(source,target,1,stride=stride,bias=False)
    def forward(self,value):return F.silu(self.main(value)+self.skip(value))


class PhoneticStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral=nn.Sequential(nn.Conv2d(1,32,5,padding=2,stride=(2,1),bias=False),nn.GroupNorm(8,32),nn.SiLU(),
                                    Block(32,48,(2,2)),Block(48,64,(2,1)))
        self.project=nn.Sequential(nn.Conv1d(640,128,1),nn.GroupNorm(8,128),nn.SiLU())
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(128,128,5,padding=2*d,dilation=d,groups=128),
                                                nn.Conv1d(128,128,1),nn.GroupNorm(8,128),nn.SiLU(),nn.Dropout(.1)) for d in [1,2,4,8]])
        self.head=nn.Conv1d(128,6,1)
    def forward(self,value):
        value=self.project(self.spectral(value).flatten(1,2))
        for block in self.temporal:value=value+block(value)
        return self.head(F.interpolate(value,size=99,mode='linear',align_corners=False)).transpose(1,2)


class TinyPhoneticStudent(nn.Module):
    """Pretrained multilingual acoustics, six phones only; no text decoder."""
    def __init__(self,directory,train_layers=1):
        super().__init__()
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        config=WhisperConfig.from_json_file(str(directory/'encoder_config.json'))
        config._attn_implementation='eager'
        self.encoder=WhisperEncoder(config)
        self.encoder.load_state_dict(torch.load(directory/'encoder_state.pt',map_location='cpu',weights_only=True))
        self.encoder.requires_grad_(False)
        if not 0<=train_layers<=len(self.encoder.layers):raise ValueError('Invalid number of encoder layers')
        if train_layers:
            for layer in self.encoder.layers[-train_layers:]:layer.requires_grad_(True)
        self.project=nn.Sequential(nn.Conv1d(config.d_model,128,1),nn.GroupNorm(8,128),nn.SiLU())
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(128,128,5,padding=2*d,dilation=d,groups=128),
                                                nn.Conv1d(128,128,1),nn.GroupNorm(8,128),nn.SiLU(),nn.Dropout(.1)) for d in [1,2,4,8]])
        self.head=nn.Conv1d(128,6,1)
    def forward(self,value):
        value=self.encoder(value.squeeze(1),return_dict=False)[0].transpose(1,2)
        value=self.project(value)
        for block in self.temporal:value=value+block(value)
        return self.head(F.interpolate(value,size=99,mode='linear',align_corners=False)).transpose(1,2)


def make_student(config,source):
    if config.get('backbone','cnn')=='tiny':return TinyPhoneticStudent(source/'whisper-encoder',config['encoder_train_layers'])
    return PhoneticStudent()


def augment_features(value):
    """Mild frequency warp simulates vocal-tract/microphone variation, preserves time."""
    batch,_,frequencies,frames=value.shape
    y,x=torch.meshgrid(torch.linspace(-1,1,frequencies,device=value.device),torch.linspace(-1,1,frames,device=value.device),indexing='ij')
    scale=torch.empty(batch,1,1,device=value.device).uniform_(.9,1.1)
    shift=torch.empty(batch,1,1,device=value.device).uniform_(-.025,.025)
    grid=torch.stack([x.expand(batch,-1,-1),y[None]*scale+shift],-1)
    return F.grid_sample(value,grid,padding_mode='border',align_corners=True)
