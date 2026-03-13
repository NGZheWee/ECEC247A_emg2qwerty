import subprocess

cmd = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",
    "+trainer.gradient_clip_val=1.0",

    # 500 Hz simulated sampling
    "datamodule._target_=emg2qwerty.lightning.DownsampledWindowedEMGDataModule",
    "+datamodule.downsample_factor=4",
    "+datamodule.use_mean_pool=true",

    # keep temporal jitter roughly similar in real time
    "temporal_jitter.max_offset=30",

    # match preprocessing to 500 Hz better
    "logspec.n_fft=16",
    "logspec.hop_length=4",

    # CNN-BiGRU with corrected in_features
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
    "module.in_features=144",
    "module.mlp_features=[384]",
    "module.block_channels=[24,24,24,24]",
    "module.kernel_width=32",
    "+module.gru_hidden_size=256",
    "+module.gru_num_layers=2",
    "+module.gru_dropout=0.1",
    "+module.bidirectional=true",

    # more conservative optimization
    "optimizer.lr=0.0003",
]

print("=" * 120)
print("Running: bigru_sampling_500hz_rescue")
print("Command:")
print(" ".join(cmd))
print("=" * 120)

subprocess.run(cmd, check=True)