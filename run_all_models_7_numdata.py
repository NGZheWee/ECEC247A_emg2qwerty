import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",

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
        "name": "bigru_data_75pct",
        "overrides": [
            "+datamodule.train_fraction=0.75",
        ],
    },
    {
        "name": "bigru_data_50pct",
        "overrides": [
            "+datamodule.train_fraction=0.50",
        ],
    },
    {
        "name": "bigru_data_25pct",
        "overrides": [
            "+datamodule.train_fraction=0.25",
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