import subprocess

COMMON_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",

    # CNN-BiGRU
    "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
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
        "name": "bigru_preproc_baseline_logspec64_16",
        "overrides": [
            "logspec.n_fft=64",
            "logspec.hop_length=16",
            "module.in_features=528",
        ],
    },
    {
        "name": "bigru_preproc_finer_time_logspec32_8",
        "overrides": [
            "logspec.n_fft=32",
            "logspec.hop_length=8",
            "module.in_features=272",
        ],
    },
    {
        "name": "bigru_preproc_finer_freq_logspec128_32",
        "overrides": [
            "logspec.n_fft=128",
            "logspec.hop_length=32",
            "module.in_features=1040",
        ],
    },
]

for exp in EXPERIMENTS:
    cmd = COMMON_CMD + exp["overrides"]
    print("=" * 120)
    print("Running:", exp["name"])
    print("Command:")
    print(" ".join(cmd))
    print("=" * 120)
    subprocess.run(cmd, check=True)