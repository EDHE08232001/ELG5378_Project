"""Interactive entry point for training or evaluating the project."""

from evaluate import evaluate_model
from train import train_model


# ------------------------
# CLI Entry Point
# ------------------------
def main() -> None:
    """Ask user whether to train or evaluate and execute the selected pipeline."""
    print("Select an option:")
    print("1) training")
    print("2) evaluating")
    selection = input("Enter 1 or 2: ").strip()

    if selection == "1":
        checkpoint_path = train_model()
        print(f"Training complete. Best checkpoint: {checkpoint_path}")
    elif selection == "2":
        summary_path = evaluate_model()
        print(f"Evaluation complete. Summary file: {summary_path}")
    else:
        print("Invalid input. Please run again and choose 1 or 2.")


if __name__ == "__main__":
    main()
