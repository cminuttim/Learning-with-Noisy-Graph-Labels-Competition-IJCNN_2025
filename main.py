import sys
sys.path.append('./source/')

import argparse
from config import ModelConfig
from trainer import ModelTrainer
from data_loader import load_dataset
import pandas as pd
import torch
import logging
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Neural Network Training and Inference')
    parser.add_argument('--test_path', help='Path to test.json.gz file')
    parser.add_argument('--train_path', help='Path to train.json.gz file (optional)')
    parser.add_argument('--model_paths_file', help='Path to file containing list of model paths for prediction (optional)')
    parser.add_argument('--num_cycles', help='Number of cycles', default=10, type=int)
    parser.add_argument('--pretrain_paths', help='Path to pretrained models')
    
    return parser.parse_args()

def main():
    args = parse_args()
    config = ModelConfig(
        test_path=args.test_path,
        train_path=args.train_path,
        num_cycles=args.num_cycles,
        pretrain_paths=args.pretrain_paths
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = ModelTrainer(config, device)
    
    if args.train_path:
        # Training mode
        logging.info(f"Starting training using {args.train_path}")
        df_train = load_dataset(args.train_path)
        trainer.train_multiple_cycles(df_train, args.num_cycles)
    else:
        # Prediction mode
        if args.model_paths_file:
            # Use the provided model paths file
            model_paths_file = args.model_paths_file
        else:
            # Look for the default model paths file
            model_paths_file = f"model_paths_{config.folder_name}.txt"
        
        if os.path.exists(model_paths_file):
            # Load model paths from the file
            with open(model_paths_file, 'r') as f:
                model_paths = [line.strip() for line in f.readlines()]
            trainer.models = model_paths
            logging.info(f"Loaded {len(model_paths)} models from {model_paths_file}")
        else:
            raise FileNotFoundError(
                f"Model paths file '{model_paths_file}' not found. "
                "Either provide --train_path for training or --model_paths_file for prediction."
            )
    
    if args.test_path:
        logging.info(f"Generating predictions for {args.test_path}")
        df_test = load_dataset(args.test_path)
        predictions, _ = trainer.predict_with_ensemble_score(df_test)
        
        # Save predictions
        output_path = f"submission/testset_{config.folder_name}.csv"
        pd.DataFrame({
            "id": range(len(predictions)),
            "pred": predictions
        }).to_csv(output_path, index=False)
        logging.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
