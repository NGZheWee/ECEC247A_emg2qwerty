import subprocess
import sys


EXPERIMENTS = [
    [
        "python", "-m", "emg2qwerty.train",
        'user="single_user"',
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=45",
        "module._target_=emg2qwerty.lightning.TDSConvLSTMCTCModule",
        "module.in_features=528",
        "module.mlp_features=[384]",
        "module.block_channels=[24,24,24,24]",
        "module.kernel_width=32",
        "+module.lstm_hidden_size=256",
        "+module.lstm_num_layers=2",
        "+module.lstm_dropout=0.2",
        "+module.bidirectional=false",
    ],
    [
        "python", "-m", "emg2qwerty.train",
        'user="single_user"',
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=45",
        "module._target_=emg2qwerty.lightning.TDSConvLSTMCTCModule",
        "module.in_features=528",
        "module.mlp_features=[384]",
        "module.block_channels=[24,24,24,24]",
        "module.kernel_width=32",
        "+module.lstm_hidden_size=256",
        "+module.lstm_num_layers=2",
        "+module.lstm_dropout=0.2",
        "+module.bidirectional=true",
    ],
    [
        "python", "-m", "emg2qwerty.train",
        'user="single_user"',
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=45",
        "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
        "module.in_features=528",
        "module.mlp_features=[384]",
        "module.block_channels=[24,24,24,24]",
        "module.kernel_width=32",
        "+module.gru_hidden_size=256",
        "+module.gru_num_layers=2",
        "+module.gru_dropout=0.2",
        "+module.bidirectional=false",
    ],
    [
        "python", "-m", "emg2qwerty.train",
        'user="single_user"',
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=45",
        "module._target_=emg2qwerty.lightning.TDSConvGRUCTCModule",
        "module.in_features=528",
        "module.mlp_features=[384]",
        "module.block_channels=[24,24,24,24]",
        "module.kernel_width=32",
        "+module.gru_hidden_size=256",
        "+module.gru_num_layers=2",
        "+module.gru_dropout=0.2",
        "+module.bidirectional=true",
    ],
]


def main() -> None:
    for i, cmd in enumerate(EXPERIMENTS, start=1):
        print("\n" + "=" * 100)
        print(f"Starting experiment {i}/{len(EXPERIMENTS)}")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 100 + "\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nExperiment {i} failed with return code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\nAll experiments finished successfully.")


if __name__ == "__main__":
    main()