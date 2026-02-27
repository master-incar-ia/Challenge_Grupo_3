from pathlib import Path
import subprocess
import sys


def run_step(script_path: Path) -> None:
    print(f"\n[PIPELINE] Running: {script_path.name}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step failed: {script_path.name} (exit code {result.returncode})"
        )
    print(f"[PIPELINE] Completed: {script_path.name}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    steps = [
        base_dir / "dataset.py",
        base_dir / "model.py",
        base_dir / "train.py",
        base_dir / "evaluate.py",
    ]

    print("[PIPELINE] Starting execution...")
    print(f"[PIPELINE] Python: {sys.executable}")

    for step in steps:
        run_step(step)

    print("\n[PIPELINE] All steps completed successfully.")


if __name__ == "__main__":
    main()
