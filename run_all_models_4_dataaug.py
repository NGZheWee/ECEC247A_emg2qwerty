import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",

    # swap datamodule
    "datamodule._target_=emg2qwerty.lightning.AugmentedWindowedEMGDataModule",

    # swap module
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
    "module.in_features=528",
    "module.mlp_features=[384]",
    "module.block_channels=[24,24,24,24]",
    "module.kernel_width=32",

    # these are NEW keys relative to baseline TDSConvCTCModule config
    "+module.gru_hidden_size=256",
    "+module.gru_num_layers=2",
    "+module.gru_dropout=0.1",
    "+module.bidirectional=true",
]

EXPERIMENTS = [
    {
        "name": "bigru_aug_noise_amp",
        "overrides": [
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.02",
            "+datamodule.num_time_masks=0",
            "+datamodule.max_time_mask_width=0",
            "+datamodule.num_freq_masks=0",
            "+datamodule.max_freq_mask_width=0",
            "+datamodule.channel_dropout_prob=0.0",
        ],
    },
    {
        "name": "bigru_aug_time_freq_mask",
        "overrides": [
            "+datamodule.amp_scale_min=1.0",
            "+datamodule.amp_scale_max=1.0",
            "+datamodule.gaussian_noise_std=0.0",
            "+datamodule.num_time_masks=2",
            "+datamodule.max_time_mask_width=12",
            "+datamodule.num_freq_masks=2",
            "+datamodule.max_freq_mask_width=4",
            "+datamodule.channel_dropout_prob=0.0",
        ],
    },
    {
        "name": "bigru_aug_channel_dropout",
        "overrides": [
            "+datamodule.amp_scale_min=1.0",
            "+datamodule.amp_scale_max=1.0",
            "+datamodule.gaussian_noise_std=0.0",
            "+datamodule.num_time_masks=0",
            "+datamodule.max_time_mask_width=0",
            "+datamodule.num_freq_masks=0",
            "+datamodule.max_freq_mask_width=0",
            "+datamodule.channel_dropout_prob=0.05",
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
    subprocess.run(cmd, check=True)