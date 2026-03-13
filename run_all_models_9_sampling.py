import subprocess

BASE_CMD = [
    "python",
    "-m",
    "emg2qwerty.train",
    "user=single_user",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=45",
    "+trainer.gradient_clip_val=1.0",

    # sampling-rate datamodule
    "datamodule._target_=emg2qwerty.lightning.DownsampledWindowedEMGDataModule",

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
        "name": "bigru_sampling_1000hz",
        "overrides": [
            "+datamodule.downsample_factor=2",
            "+datamodule.use_mean_pool=true",
            "temporal_jitter.max_offset=60",
            "optimizer.lr=0.001",
        ],
    },
    {
        "name": "bigru_sampling_500hz",
        "overrides": [
            "+datamodule.downsample_factor=4",
            "+datamodule.use_mean_pool=true",
            "temporal_jitter.max_offset=30",
            "optimizer.lr=0.0003",
        ],
    },
    {
        "name": "bigru_sampling_250hz",
        "overrides": [
            "+datamodule.downsample_factor=8",
            "+datamodule.use_mean_pool=true",
            "temporal_jitter.max_offset=15",
            "optimizer.lr=0.0001",
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