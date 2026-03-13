import subprocess

COMMON_PREFIX = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",

    # datamodule that subsamples train sessions
    "datamodule._target_=emg2qwerty.lightning.DataFractionWindowedEMGDataModule",

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
        "name": "bigru_data_75pct_matched_steps",
        "epochs": 60,   # 45 / 0.75
        "fraction": 0.75,
    },
    {
        "name": "bigru_data_50pct_matched_steps",
        "epochs": 90,   # 45 / 0.50
        "fraction": 0.50,
    },
    {
        "name": "bigru_data_25pct_matched_steps",
        "epochs": 180,  # 45 / 0.25
        "fraction": 0.25,
    },
]

for exp in EXPERIMENTS:
    cmd = COMMON_PREFIX + [
        f"trainer.max_epochs={exp['epochs']}",
        f"+datamodule.train_fraction={exp['fraction']}",
    ]

    print("=" * 120)
    print("Running:", exp["name"])
    print("Command:")
    print(" ".join(cmd))
    print("=" * 120)

    subprocess.run(cmd, check=True)