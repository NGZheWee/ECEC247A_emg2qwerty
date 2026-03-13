# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

# =========================================================
# BY DERRICK: augmentation wrapper + augmented datamodule
# =========================================================

import random
import torch


class AugmentedPostTransform:
    """
    Wrap an existing transform and then apply additional training-only
    spectrogram-level augmentations.

    Expected tensor shape after base_transform:
        (T, B, C, F)
    where:
        T = time
        B = bands (2)
        C = electrode channels (16)
        F = frequency bins
    """

    def __init__(
        self,
        base_transform,
        amp_scale_min: float = 1.0,
        amp_scale_max: float = 1.0,
        gaussian_noise_std: float = 0.0,
        num_time_masks: int = 0,
        max_time_mask_width: int = 0,
        num_freq_masks: int = 0,
        max_freq_mask_width: int = 0,
        channel_dropout_prob: float = 0.0,
    ) -> None:
        self.base_transform = base_transform
        self.amp_scale_min = amp_scale_min
        self.amp_scale_max = amp_scale_max
        self.gaussian_noise_std = gaussian_noise_std
        self.num_time_masks = num_time_masks
        self.max_time_mask_width = max_time_mask_width
        self.num_freq_masks = num_freq_masks
        self.max_freq_mask_width = max_freq_mask_width
        self.channel_dropout_prob = channel_dropout_prob

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.amp_scale_min == 1.0 and self.amp_scale_max == 1.0:
            return x
        scale = random.uniform(self.amp_scale_min, self.amp_scale_max)
        return x * scale

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.gaussian_noise_std <= 0:
            return x
        ref_std = x.std().clamp_min(1e-6)
        noise = torch.randn_like(x) * (self.gaussian_noise_std * ref_std)
        return x + noise

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_time_masks <= 0 or self.max_time_mask_width <= 0:
            return x

        T = x.shape[0]
        if T <= 1:
            return x

        for _ in range(self.num_time_masks):
            w = random.randint(1, min(self.max_time_mask_width, T))
            if w >= T:
                continue
            t0 = random.randint(0, T - w)
            x[t0:t0 + w, :, :, :] = 0
        return x

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freq_masks <= 0 or self.max_freq_mask_width <= 0:
            return x

        F = x.shape[-1]
        if F <= 1:
            return x

        for _ in range(self.num_freq_masks):
            w = random.randint(1, min(self.max_freq_mask_width, F))
            if w >= F:
                continue
            f0 = random.randint(0, F - w)
            x[:, :, :, f0:f0 + w] = 0
        return x

    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_dropout_prob <= 0:
            return x

        # x shape: (T, B, C, F)
        B = x.shape[1]
        C = x.shape[2]

        for b in range(B):
            for c in range(C):
                if random.random() < self.channel_dropout_prob:
                    x[:, b, c, :] = 0
        return x

    def __call__(self, sample):
        x = self.base_transform(sample)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = x.clone()

        x = self._amplitude_scale(x)
        x = self._gaussian_noise(x)
        x = self._time_mask(x)
        x = self._freq_mask(x)
        x = self._channel_dropout(x)

        return x


class AugmentedWindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform,
        val_transform,
        test_transform,
        amp_scale_min: float = 1.0,
        amp_scale_max: float = 1.0,
        gaussian_noise_std: float = 0.0,
        num_time_masks: int = 0,
        max_time_mask_width: int = 0,
        num_freq_masks: int = 0,
        max_freq_mask_width: int = 0,
        channel_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.amp_scale_min = amp_scale_min
        self.amp_scale_max = amp_scale_max
        self.gaussian_noise_std = gaussian_noise_std
        self.num_time_masks = num_time_masks
        self.max_time_mask_width = max_time_mask_width
        self.num_freq_masks = num_freq_masks
        self.max_freq_mask_width = max_freq_mask_width
        self.channel_dropout_prob = channel_dropout_prob

    def setup(self, stage: str | None = None) -> None:
        augmented_train_transform = AugmentedPostTransform(
            base_transform=self.train_transform,
            amp_scale_min=self.amp_scale_min,
            amp_scale_max=self.amp_scale_max,
            gaussian_noise_std=self.gaussian_noise_std,
            num_time_masks=self.num_time_masks,
            max_time_mask_width=self.max_time_mask_width,
            num_freq_masks=self.num_freq_masks,
            max_freq_mask_width=self.max_freq_mask_width,
            channel_dropout_prob=self.channel_dropout_prob,
        )

        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=augmented_train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


# =========================================================
# BY DERRICK: fixed channel subset masking for Question 3
# =========================================================


class FixedChannelMaskPostTransform:
    """
    Apply base_transform first, then keep only a fixed subset of channels
    per band and zero out the rest.

    Expected tensor shape after base_transform:
        (T, B, C, F)
    where:
        T = time
        B = bands (2)
        C = channels per band (16)
        F = frequency bins
    """

    def __init__(self, base_transform, keep_channels_per_band: int) -> None:
        self.base_transform = base_transform
        self.keep_channels_per_band = keep_channels_per_band

    @staticmethod
    def _evenly_spaced_indices(total_channels: int, keep_channels: int) -> list[int]:
        if keep_channels >= total_channels:
            return list(range(total_channels))
        if keep_channels <= 0:
            return []

        # Evenly spread indices across the channel axis
        idx = np.linspace(0, total_channels - 1, num=keep_channels)
        idx = np.round(idx).astype(int).tolist()

        # Remove duplicates while preserving order
        dedup = []
        seen = set()
        for i in idx:
            if i not in seen:
                dedup.append(i)
                seen.add(i)

        # If rounding caused duplicates and we have too few, fill remaining
        if len(dedup) < keep_channels:
            for i in range(total_channels):
                if i not in seen:
                    dedup.append(i)
                    seen.add(i)
                if len(dedup) == keep_channels:
                    break

        return dedup

    def __call__(self, sample):
        x = self.base_transform(sample)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        x = x.clone()

        # x shape: (T, B, C, F)
        total_channels = x.shape[2]
        keep_idx = self._evenly_spaced_indices(
            total_channels=total_channels,
            keep_channels=self.keep_channels_per_band,
        )

        mask = torch.zeros(total_channels, dtype=x.dtype, device=x.device)
        if len(keep_idx) > 0:
            mask[keep_idx] = 1.0

        x = x * mask.view(1, 1, total_channels, 1)
        return x


class ChannelAblationWindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform,
        val_transform,
        test_transform,
        keep_channels_per_band: int = 16,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.keep_channels_per_band = keep_channels_per_band

    def setup(self, stage: str | None = None) -> None:
        masked_train_transform = FixedChannelMaskPostTransform(
            base_transform=self.train_transform,
            keep_channels_per_band=self.keep_channels_per_band,
        )
        masked_val_transform = FixedChannelMaskPostTransform(
            base_transform=self.val_transform,
            keep_channels_per_band=self.keep_channels_per_band,
        )
        masked_test_transform = FixedChannelMaskPostTransform(
            base_transform=self.test_transform,
            keep_channels_per_band=self.keep_channels_per_band,
        )

        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=masked_train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=masked_val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=masked_test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

# BY DERRICK
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = self._build_pe(max_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, d_model)
        T = x.size(0)

        if T > self.pe.size(0):
            pe = self._build_pe(T, self.d_model).to(device=x.device, dtype=x.dtype)
        else:
            pe = self.pe[:T].to(dtype=x.dtype)

        x = x + pe
        return self.dropout(x)


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

# =========================================================
# BY DERRICK: CNN-LSTM
# =========================================================
class TDSConvLSTMCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        rnn_directions = 2 if bidirectional else 1

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
        )

        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_directions * lstm_hidden_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)   # (T, N, num_features)
        x, _ = self.rnn(x)          # (T, N, rnn_directions * hidden_size)
        x = self.classifier(x)      # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvGRUCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        gru_dropout: float,
        bidirectional: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        rnn_directions = 2 if bidirectional else 1

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
        )

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_directions * gru_hidden_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)   # (T, N, num_features)
        x, _ = self.rnn(x)          # (T, N, rnn_directions * hidden_size)
        x = self.classifier(x)      # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvTransformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
        )

        self.input_proj = nn.Linear(num_features, transformer_d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=transformer_d_model,
            dropout=transformer_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)       # (T, N, num_features)
        x = self.input_proj(x)          # (T, N, d_model)
        x = self.pos_encoder(x)         # (T, N, d_model)
        x = self.transformer(x)         # (T, N, d_model)
        x = self.classifier(x)          # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

# =========================================================
# BY DERRICK: Transformer
# =========================================================
class TransformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_dim_feedforward: int,
        transformer_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Pure transformer front-end: no TDSConvEncoder
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),  # (T, N, num_features)
        )

        self.input_proj = nn.Linear(num_features, transformer_d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=transformer_d_model,
            dropout=transformer_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)       # (T, N, num_features)
        x = self.input_proj(x)          # (T, N, d_model)
        x = self.pos_encoder(x)         # (T, N, d_model)
        x = self.transformer(x)         # (T, N, d_model)
        x = self.classifier(x)          # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # No temporal shrinkage here, since there is no TDS conv encoder
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# =========================================================
# BY DERRICK: training-data fraction datamodule
# =========================================================
class DataFractionWindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform,
        val_transform,
        test_transform,
        train_fraction: float = 1.0,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = list(train_sessions)
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.train_fraction = train_fraction

    @staticmethod
    def _select_evenly_spaced_subset(
        sessions: list[Path],
        fraction: float,
    ) -> list[Path]:
        if fraction >= 1.0:
            return sessions
        if fraction <= 0.0:
            raise ValueError("train_fraction must be > 0")

        total = len(sessions)
        keep = max(1, int(round(total * fraction)))

        if keep >= total:
            return sessions

        # Evenly spaced deterministic subset to preserve coverage across sessions
        idx = np.linspace(0, total - 1, num=keep)
        idx = np.round(idx).astype(int).tolist()

        dedup = []
        seen = set()
        for i in idx:
            if i not in seen:
                dedup.append(i)
                seen.add(i)

        if len(dedup) < keep:
            for i in range(total):
                if i not in seen:
                    dedup.append(i)
                    seen.add(i)
                if len(dedup) == keep:
                    break

        return [sessions[i] for i in dedup]

    def setup(self, stage: str | None = None) -> None:
        selected_train_sessions = self._select_evenly_spaced_subset(
            sessions=self.train_sessions,
            fraction=self.train_fraction,
        )

        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in selected_train_sessions
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

import copy


# =========================================================
# BY DERRICK: raw EMG downsampling for sampling-rate study
# =========================================================

class RawEMGDownsampleTransform:
    """
    Downsample raw EMG arrays before the existing transform pipeline.

    This simulates a lower effective sampling rate while keeping the rest
    of the model pipeline unchanged.

    Expected sample fields include:
        sample["emg_left"], sample["emg_right"]
    """

    def __init__(
        self,
        base_transform,
        downsample_factor: int = 1,
        use_mean_pool: bool = True,
    ) -> None:
        self.base_transform = base_transform
        self.downsample_factor = downsample_factor
        self.use_mean_pool = use_mean_pool

    @staticmethod
    def _downsample_array(x: np.ndarray, factor: int, use_mean_pool: bool) -> np.ndarray:
        if factor <= 1:
            return x

        if use_mean_pool:
            usable = (x.shape[0] // factor) * factor
            if usable < factor:
                return x[::factor]
            x_trim = x[:usable]
            new_shape = (usable // factor, factor) + x.shape[1:]
            return x_trim.reshape(new_shape).mean(axis=1)
        else:
            return x[::factor]

    def __call__(self, sample):
        if self.downsample_factor <= 1:
            return self.base_transform(sample)

        emg_left = np.asarray(sample["emg_left"])
        emg_right = np.asarray(sample["emg_right"])

        downsampled_sample = {
            "emg_left": self._downsample_array(
                emg_left,
                self.downsample_factor,
                self.use_mean_pool,
            ),
            "emg_right": self._downsample_array(
                emg_right,
                self.downsample_factor,
                self.use_mean_pool,
            ),
        }

        return self.base_transform(downsampled_sample)

        
class DownsampledWindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform,
        val_transform,
        test_transform,
        downsample_factor: int = 1,
        use_mean_pool: bool = True,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.downsample_factor = downsample_factor
        self.use_mean_pool = use_mean_pool

    def setup(self, stage: str | None = None) -> None:
        wrapped_train_transform = RawEMGDownsampleTransform(
            base_transform=self.train_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )
        wrapped_val_transform = RawEMGDownsampleTransform(
            base_transform=self.val_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )
        wrapped_test_transform = RawEMGDownsampleTransform(
            base_transform=self.test_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )

        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=wrapped_train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=wrapped_val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=wrapped_test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


# =========================================================
# BY DERRICK: combined downsampling + augmentation datamodule
# =========================================================

class DownsampledAugmentedWindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform,
        val_transform,
        test_transform,
        downsample_factor: int = 1,
        use_mean_pool: bool = True,
        amp_scale_min: float = 1.0,
        amp_scale_max: float = 1.0,
        gaussian_noise_std: float = 0.0,
        num_time_masks: int = 0,
        max_time_mask_width: int = 0,
        num_freq_masks: int = 0,
        max_freq_mask_width: int = 0,
        channel_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.downsample_factor = downsample_factor
        self.use_mean_pool = use_mean_pool

        self.amp_scale_min = amp_scale_min
        self.amp_scale_max = amp_scale_max
        self.gaussian_noise_std = gaussian_noise_std
        self.num_time_masks = num_time_masks
        self.max_time_mask_width = max_time_mask_width
        self.num_freq_masks = num_freq_masks
        self.max_freq_mask_width = max_freq_mask_width
        self.channel_dropout_prob = channel_dropout_prob

    def setup(self, stage: str | None = None) -> None:
        # First downsample raw EMG, then apply base transform, then apply
        # spectrogram-level augmentation on the training path only.
        downsampled_train_transform = RawEMGDownsampleTransform(
            base_transform=self.train_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )
        augmented_train_transform = AugmentedPostTransform(
            base_transform=downsampled_train_transform,
            amp_scale_min=self.amp_scale_min,
            amp_scale_max=self.amp_scale_max,
            gaussian_noise_std=self.gaussian_noise_std,
            num_time_masks=self.num_time_masks,
            max_time_mask_width=self.max_time_mask_width,
            num_freq_masks=self.num_freq_masks,
            max_freq_mask_width=self.max_freq_mask_width,
            channel_dropout_prob=self.channel_dropout_prob,
        )

        downsampled_val_transform = RawEMGDownsampleTransform(
            base_transform=self.val_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )
        downsampled_test_transform = RawEMGDownsampleTransform(
            base_transform=self.test_transform,
            downsample_factor=self.downsample_factor,
            use_mean_pool=self.use_mean_pool,
        )

        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=augmented_train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )

        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=downsampled_val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )

        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=downsampled_test_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )