"""Small binary CNN: no tokenizer, vocabulary, ASR model or transcript output."""
import torch
from torch import nn


class Separable(nn.Module):
    def __init__(self, source, target, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(source, source, 3, stride=stride, padding=1, groups=source, bias=False),
                                    nn.Conv2d(source, target, 1, bias=False), nn.GroupNorm(8, target), nn.ReLU())

    def forward(self, value):
        return self.layers(value)


class WakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
                                     nn.GroupNorm(8, 24), nn.ReLU(), Separable(24, 48, 2),
                                     Separable(48, 64, 2), Separable(64, 96), Separable(96, 96))
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(96, 1))

    def forward(self, value):
        return self.head(self.encoder(value).mean(dim=(2, 3))).squeeze(-1)


class ProbabilityModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value):
        return torch.sigmoid(self.model(value))


class Residual(nn.Module):
    def __init__(self, source, target, stride=(1, 1)):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(source, target, 3, stride=stride, padding=1, bias=False),
                                  nn.GroupNorm(8, target), nn.SiLU(),
                                  nn.Conv2d(target, target, 3, padding=1, bias=False), nn.GroupNorm(8, target))
        self.skip = nn.Identity() if source == target and stride == (1, 1) else nn.Conv2d(source, target, 1, stride=stride, bias=False)

    def forward(self, value):
        return nn.functional.silu(self.main(value) + self.skip(value))


class TemporalWakeModel(nn.Module):
    """Retain frequency position and temporal order; attention and max pool over time."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 24, 5, stride=(2, 2), padding=2, bias=False),
                                     nn.GroupNorm(8, 24), nn.SiLU(), Residual(24, 32, (2, 1)),
                                     Residual(32, 48, (2, 1)), Residual(48, 64))
        self.project = nn.Sequential(nn.Conv1d(64 * 5, 96, 1), nn.GroupNorm(8, 96), nn.SiLU())
        self.temporal = nn.ModuleList([nn.Sequential(nn.Conv1d(96, 96, 5, dilation=d, padding=2*d, groups=96),
                                       nn.Conv1d(96, 96, 1), nn.GroupNorm(8, 96), nn.SiLU(), nn.Dropout(.1)) for d in [1, 2, 4]])
        self.attention = nn.Conv1d(96, 1, 1)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, value):
        value = self.encoder(value)
        value = self.project(value.flatten(1, 2))
        for block in self.temporal: value = value + block(value)
        weights = torch.softmax(self.attention(value), dim=-1)
        pooled = torch.cat([(value * weights).sum(-1), value.amax(-1)], dim=1)
        return self.head(pooled).squeeze(-1)


def make_model(config):
    architecture = config.get('architecture', 'legacy')
    if architecture == 'legacy': return WakeModel()
    if architecture == 'temporal_v2': return TemporalWakeModel()
    if architecture == 'speech_embedding_v3': return EmbeddingWakeModel()
    if architecture == 'whisper_encoder_v4': return WhisperWakeModel(config)
    raise ValueError(f'Unknown architecture: {architecture}')


class EmbeddingWakeModel(nn.Module):
    """Learn a keyword head on frozen speech embeddings, retaining context order."""
    def __init__(self):
        super().__init__()
        self.normalization = nn.LayerNorm(96)
        self.project = nn.Sequential(nn.Conv1d(96, 128, 1), nn.SiLU(), nn.Dropout(.15))
        self.temporal = nn.ModuleList([nn.Sequential(nn.Conv1d(128, 128, 3, padding=d, dilation=d, groups=128),
                                    nn.Conv1d(128, 128, 1), nn.GroupNorm(8, 128), nn.SiLU(), nn.Dropout(.15)) for d in [1, 2, 4]])
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.25), nn.Linear(128 * 16, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, value):
        value = self.normalization(value.squeeze(1).transpose(1, 2)).transpose(1, 2)
        value = self.project(value)
        for block in self.temporal: value = value + block(value)
        return self.head(value).squeeze(-1)


class WhisperWakeModel(nn.Module):
    """Multilingual acoustic encoder plus a binary head; no decoder or text output."""
    def __init__(self, config):
        super().__init__()
        from pathlib import Path
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        directory = Path(config.get('encoder_dir') or Path(config['work_dir']) / 'whisper-encoder')
        encoder_config = WhisperConfig.from_json_file(str(directory / 'encoder_config.json'))
        encoder_config._attn_implementation = 'eager'
        self.encoder = WhisperEncoder(encoder_config)
        self.encoder.load_state_dict(torch.load(directory / 'encoder_state.pt', map_location='cpu', weights_only=True))
        self.encoder.requires_grad_(False)
        layers = config.get('encoder_train_layers', 2)
        if not isinstance(layers, int) or not 0 <= layers <= len(self.encoder.layers):
            raise ValueError(f'encoder_train_layers must be between 0 and {len(self.encoder.layers)}')
        if layers:
            for layer in self.encoder.layers[-layers:]: layer.requires_grad_(True)
        self.project = nn.Sequential(nn.Conv1d(encoder_config.d_model, 96, 1), nn.SiLU(), nn.Dropout(.15))
        self.attention = nn.Conv1d(96, 1, 1)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, value):
        value = self.encoder(value.squeeze(1), return_dict=False)[0].transpose(1, 2)
        value = self.project(value)
        attention = torch.softmax(self.attention(value), dim=-1)
        return self.head(torch.cat([(value * attention).sum(-1), value.amax(-1)], dim=1)).squeeze(-1)


class EnsembleLogits(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, value):
        return torch.stack([model(value) for model in self.models]).mean(0)


def spec_augment(value):
    # Small masks cannot remove the entire short wake word.
    batch, _, frequencies, frames = value.shape
    if frames == 16:
        # Speech embedding axes are latent features, not mel frequencies.
        return value + torch.randn_like(value) * .015
    time_start = torch.randint(frames - 8, (batch, 1, 1, 1), device=value.device)
    frequency_start = torch.randint(frequencies - 4, (batch, 1, 1, 1), device=value.device)
    time = torch.arange(frames, device=value.device)[None, None, None, :]
    frequency = torch.arange(frequencies, device=value.device)[None, None, :, None]
    mask = ((time >= time_start) & (time < time_start + 8)) | ((frequency >= frequency_start) & (frequency < frequency_start + 4))
    return value.masked_fill(mask, 0)
