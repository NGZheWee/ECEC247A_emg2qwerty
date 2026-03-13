import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",

    # augmented datamodule already added in lightning.py
    "datamodule._target_=emg2qwerty.lightning.AugmentedWindowedEMGDataModule",

    # CNN-BiGRU backbone
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
    "module.in_features=528",
    "module.mlp_features=[384]",
    "module.block_channels=[24,24,24,24]",
    "module.kernel_width=32",
    "+module.gru_hidden_size=256",
    "+module.gru_num_layers=2",
    "+module.gru_dropout=0.1",
    "+module.bidirectional=true",

    # no masking-based augmentation in this sweep
    "+datamodule.num_time_masks=0",
    "+datamodule.max_time_mask_width=0",
    "+datamodule.num_freq_masks=0",
    "+datamodule.max_freq_mask_width=0",
]

EXPERIMENTS = [
    {
        "name": "bigru_aug_combo_1_cd005_amp0911_noise002",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.05",
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.02",
        ],
    },
    {
        "name": "bigru_aug_combo_2_cd003_amp0911_noise002",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.03",
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.02",
        ],
    },
    {
        "name": "bigru_aug_combo_3_cd007_amp0911_noise002",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.07",
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.02",
        ],
    },
    {
        "name": "bigru_aug_combo_4_cd005_amp095105_noise001",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.05",
            "+datamodule.amp_scale_min=0.95",
            "+datamodule.amp_scale_max=1.05",
            "+datamodule.gaussian_noise_std=0.01",
        ],
    },
    {
        "name": "bigru_aug_combo_5_cd005_amp085115_noise002",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.05",
            "+datamodule.amp_scale_min=0.85",
            "+datamodule.amp_scale_max=1.15",
            "+datamodule.gaussian_noise_std=0.02",
        ],
    },
    {
        "name": "bigru_aug_combo_6_cd005_amp0911_noise003",
        "overrides": [
            "+datamodule.channel_dropout_prob=0.05",
            "+datamodule.amp_scale_min=0.9",
            "+datamodule.amp_scale_max=1.1",
            "+datamodule.gaussian_noise_std=0.03",
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