import argparse
import sys
import os
from pathlib import Path

# Add current directory to path to allow imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    from clean_data import clean_and_split
    from train import train_model
    from evaluate import evaluate_model
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the project root or backend/ml directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unified Chest Disease Prediction Pipeline")
    
    # Pipeline control flags
    parser.add_argument('--skip_clean', action='store_true', help='Skip data cleaning and splitting step')
    parser.add_argument('--skip_train', action='store_true', help='Skip training step')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation step')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    
    # Execution mode
    parser.add_argument('--quick_run', action='store_true', help='Run with reduced epochs and batch limit for testing')

    args = parser.parse_args()
    
    print("="*60)
    print("       CHEST DISEASE PREDICTION PIPELINE START")
    print("="*60)

    # Adjust settings for quick run
    max_batches = None
    if args.quick_run:
        print("[!] Quick Run Mode Enabled")
        args.epochs = 1
        max_batches = 5
        print(f"    - Epochs set to 1")
        print(f"    - Max batches set to 5")

    # Step 1: Data Cleaning
    if not args.skip_clean:
        print("\n\n" + "-"*30)
        print(">>> STEP 1: DATA CLEANING & SPLITTING")
        print("-"*30)
        try:
            clean_and_split()
        except Exception as e:
            print(f"ERROR in Data Cleaning: {e}")
            sys.exit(1)
    else:
        print("\n>>> STEP 1: Data Cleaning SKIPPED")

    # Step 2: Training
    if not args.skip_train:
        print("\n\n" + "-"*30)
        print(">>> STEP 2: MODEL TRAINING")
        print("-"*30)
        try:
            train_model(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_batches=max_batches
            )
        except Exception as e:
            print(f"ERROR in Training: {e}")
            sys.exit(1)
    else:
        print("\n>>> STEP 2: Training SKIPPED")

    # Step 3: Evaluation
    if not args.skip_eval:
        print("\n\n" + "-"*30)
        print(">>> STEP 3: MODEL EVALUATION")
        print("-"*30)
        try:
            evaluate_model(
                batch_size=args.batch_size, 
                max_batches=max_batches
            )
        except Exception as e:
            print(f"ERROR in Evaluation: {e}")
            sys.exit(1)
    else:
        print("\n>>> STEP 3: Evaluation SKIPPED")

    print("\n" + "="*60)
    print("       PIPELINE EXECUTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
