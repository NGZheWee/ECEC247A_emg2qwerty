import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "emg2qwerty.train",
        "user=single_user",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.max_epochs=45",
        "module._target_=emg2qwerty.lightning.TransformerCTCModule",
        "module.in_features=528",
        "module.mlp_features=[384]",
        "~module.block_channels",
        "~module.kernel_width",
        "+module.transformer_d_model=256",
        "+module.transformer_nhead=8",
        "+module.transformer_num_layers=4",
        "+module.transformer_dim_feedforward=1024",
        "+module.transformer_dropout=0.1",
    ]

    print("=" * 100, flush=True)
    print("Starting pure transformer training", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()