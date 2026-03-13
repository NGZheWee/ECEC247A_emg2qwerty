import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",

    # fixed-channel datamodule
    "datamodule._target_=emg2qwerty.lightning.ChannelAblationWindowedEMGDataModule",

    # CNN-BiGRU
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
    "module.in_features=528",
    "module.mlp_features=[384]",
    "module.block_channels=[24,24,24,24]",
    "module.kernel_width=32",
    "+module.gru_hidden_size=256",
    "+module.gru_num_layers=2",
    "+module.gru_dropout=0.1",
    "+module.bidirectional=true",
]

EXPERIMENTS = [
    {
        "name": "bigru_channels_24_total",
        "overrides": [
            "+datamodule.keep_channels_per_band=12",
        ],
    },
    {
        "name": "bigru_channels_16_total",
        "overrides": [
            "+datamodule.keep_channels_per_band=8",
        ],
    },
    {
        "name": "bigru_channels_8_total",
        "overrides": [
            "+datamodule.keep_channels_per_band=4",
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