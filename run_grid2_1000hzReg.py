import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",

    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=100",

    # keep the same 1000 Hz branch as your chosen reference
    "+trainer.gradient_clip_val=1.0",

    # combined downsampling + augmentation datamodule
    "datamodule._target_=emg2qwerty.lightning.DownsampledAugmentedWindowedEMGDataModule",
    "+datamodule.downsample_factor=2",
    "+datamodule.use_mean_pool=true",

    # 1000 Hz preprocessing (same as your chosen reference)
    "temporal_jitter.max_offset=60",
    "logspec.n_fft=64",
    "logspec.hop_length=16",

    # disable masking-based augmentation in both runs
    "+datamodule.num_time_masks=0",
    "+datamodule.max_time_mask_width=0",
    "+datamodule.num_freq_masks=0",
    "+datamodule.max_freq_mask_width=0",

    # BiGRU backbone
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
    "module.in_features=528",
    "module.mlp_features=[384]",
    "module.block_channels=[24,24,24,24]",
    "module.kernel_width=32",
    "+module.gru_hidden_size=256",
    "+module.gru_num_layers=2",
    "+module.gru_dropout=0.1",
    "+module.bidirectional=true",

    # optimizer setting from your 1000 Hz chosen branch
    "optimizer.lr=0.001",
]

EXPERIMENTS = [
    {
        "name": "A1_1000hz_channel_dropout_only",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.05",
            "+datamodule.amp_scale_min=1.0",
            "+datamodule.amp_scale_max=1.0",
            "+datamodule.gaussian_noise_std=0.0",
        ],
    },
    {
        "name": "A2_1000hz_amp_noise_only",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.0",
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.02",
        ],
    },
]

for exp in EXPERIMENTS:
    cmd = BASE_CMD + exp["overrides"]

    print("=" * 120)
    print("Running:", exp["name"])
    print("Command:")
    print(" ".join(cmd))
    print("=" * 120)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Experiment failed: {exp['name']} (return code {result.returncode})")
        print("Continuing to next experiment...")