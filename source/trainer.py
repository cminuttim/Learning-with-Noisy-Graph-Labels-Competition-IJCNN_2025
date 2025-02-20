# source/trainer.py
import logging
import os
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
import torch
import os

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config import ModelConfig
from models import EdgeVGAE
from utils import set_seed
from data_loader import create_dataset_from_dataframe

def warm_up_lr(epoch, num_epoch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = (epoch+1)**3 * init_lr / num_epoch_warm_up**3

class ModelTrainer:
    def __init__(self, config: 'ModelConfig', device: str):
        self.config = config
        self.device = device
        self.models: List[str] = []
        self.pretrain_models: List[str] = []
        self.best_f1_scores = []        
        self.setup_directories()
        self.setup_logging()
        self.criterion = torch.nn.CrossEntropyLoss()

    def setup_directories(self):
        directories = ['checkpoints', 'submission', 'logs']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'logs/training_{self.config.folder_name}_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
    def evaluate_model(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                z, mu, logvar, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                # Calculate CrossEntropyLoss
                loss = self.criterion(class_logits, data.y)
                
                # Get predictions
                pred_classes = class_logits.argmax(dim=1).cpu().numpy()
                true_labels = data.y.cpu().numpy()
                
                # Accumulate predictions and labels for F1 score
                all_preds.extend(pred_classes)
                all_labels.extend(true_labels)
                
                # Accumulate loss
                batch_size = data.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'cross_entropy_loss': avg_loss,
            'f1_score': f1,
            'num_samples': total_samples
        }
    
    def load_pretrained(self):
        if self.config.pretrain_paths is not None:
            if os.path.exists(self.config.pretrain_paths):
                # Load model paths from the file
                with open(self.config.pretrain_paths, 'r') as f:
                    model_paths = [line.strip() for line in f.readlines()]
                self.pretrain_models = model_paths
                logging.info(f"Loaded {len(self.pretrain_models)} pretrained models from {self.config.pretrain_paths}")
            else:
                raise FileNotFoundError(
                    f"Model paths file '{self.config.pretrain_paths}' not found. "
                )
    

    def train_single_cycle(self, cycle_num: int, train_data, val_data):
        logging.info(f"\nStarting training cycle {cycle_num}")
        
        model = EdgeVGAE(1, 7, self.config.hidden_dim, 
                        self.config.latent_dim, 
                        self.config.num_classes).to(self.device)

        # Load pretrained models if any
        if len(self.pretrain_models)>0:
            n = len(self.pretrain_models)
            model_file = self.pretrain_models[(cycle_num-1)%n]
            model_data = torch.load(model_file, map_location=torch.device(self.device))
            model.load_state_dict(model_data['model_state_dict'])
            logging.info(f"Loaded pretrained model: {model_file}")

                    
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        
        # Warm-up scheduler
        warmup_epochs = self.config.warmup  # Number of warm-up epochs        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Monitor validation 
            factor=0.7,  # Reduce LR by 50% on plateau
            patience=10,  # Number of epochs with no improvement
            min_lr=1e-6,
            verbose=True
        )
        
        best_val_loss = float('inf')
        best_f1 = 0.0  # Track best F1 score even though we're not using it for selection
        epoch_best = 0
        best_model_path = None
        
        for epoch in range(self.config.epochs):

            # Warm-up phase
            if epoch < warmup_epochs:
                warm_up_lr(epoch, warmup_epochs, self.config.learning_rate, optimizer)
                if epoch==(warmup_epochs-1):
                    logging.info("Warm-up epochs finished")
                    
            # Training
            model.train()
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler)

                
            # Validation
            val_metrics = self.evaluate_model(model, val_loader)
            val_loss = val_metrics['cross_entropy_loss']
            val_f1 = val_metrics['f1_score']
            
            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(f'Cycle {cycle_num}, Epoch {epoch+1}, LR {optimizer.param_groups[0]["lr"]:.3e}, '
                           f'Train Loss: {train_loss:.4f}, '
                           f'Val Loss: {val_loss:.4f}, '
                           f'Val F1: {val_f1:.4f}')
                
            # Update lr after warm-up (Scheduler ReduceLROnPlateau)
            if epoch >= warmup_epochs:
                scheduler.step(val_f1)
            
            # Save model if it has the best validation loss so far
            #if val_loss < best_val_loss:
            if val_f1 > best_f1:
                best_val_loss = val_loss
                best_f1 = val_f1  # Store F1 score of the best model
                epoch_best = epoch
                
                # Save the model with both metrics in filename
                best_model_path = (f"checkpoints/model_{self.config.folder_name}_"
                                 f"cycle_{cycle_num}_epoch_{epoch}_"
                                 f"loss_{val_loss:.3f}_f1_{val_f1:.3f}.pth")
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'train_loss': train_loss,
                    'config': self.config
                }, best_model_path)
                
                logging.info(f"New best model saved: {best_model_path}")
                logging.info(f"Best validation metrics - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

            # If there is no improvement, reload the best parameters
            if (epoch - epoch_best) > self.config.early_stopping_patience//2 and epoch%10==0:
                model_data = torch.load(best_model_path)
                model.load_state_dict(model_data['model_state_dict'])
                logging.info(f"Reloading best model: {best_model_path}")
                
            # Early stopping based on validation loss
            if (epoch - epoch_best) > self.config.early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.models.append(best_model_path)
        return best_val_loss, best_f1, best_model_path
    
    def train_epoch(self, model, train_loader, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            z, mu, logvar, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # Calculate losses
            recon_loss = model.recon_loss(z, data.edge_index, data.edge_attr)
            kl_loss = model.kl_loss(mu, logvar)
            class_loss = self.criterion(class_logits, data.y)
            
            # Total loss
            loss = 0.15 * recon_loss + 0.1 * kl_loss + class_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step()
            
            # Accumulate loss
            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples

    def train_multiple_cycles(self, df, num_cycles=10):

        self.load_pretrained()
        results = []
                
        for cycle in range(num_cycles):
            cycle_seed = cycle + 1
            #cycle_seed = np.random.randint(100,1000)
            set_seed(cycle_seed)
            logging.info(f"\nStarting cycle {cycle + 1} with seed {cycle_seed}")
    
            train_data, val_data = self.prepare_data_split(df, seed=cycle+1)
            val_loss, val_f1, model_path = self.train_single_cycle(cycle + 1, train_data, val_data)
    
            results.append({
                'cycle': cycle + 1,
                'seed': cycle_seed,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'model_path': model_path
            })
    
            # Log summary of this cycle
            logging.info(f"Cycle {cycle + 1} completed:")
            logging.info(f"- Final validation loss: {val_loss:.4f}")
            logging.info(f"- Final F1 score: {val_f1:.4f}")
    
            # # Keep only the top 5 models based on validation loss
            # if len(self.models) > 5:
            #     models_with_loss = [(path, self.get_model_loss(path)) for path in self.models]
            #     models_with_loss.sort(key=lambda x: x[1])
            #     self.models = [path for path, _ in models_with_loss[:5]]
    
            #     # Log the kept models
            #     logging.info("\nKept top 5 models:")
            #     for model_path in self.models:
            #         checkpoint = torch.load(model_path, map_location=self.device)
            #         logging.info(f"- {os.path.basename(model_path)}")
            #         logging.info(f"  Loss: {checkpoint['val_loss']:.4f}, F1: {checkpoint['val_f1']:.4f}")
    
        # Save the paths of the best models to a file
        model_paths_file = f"model_paths_{self.config.folder_name}.txt"
        with open(model_paths_file, 'w') as f:
            for model_path in self.models:
                f.write(f"{model_path}\n")
        logging.info(f"Saved paths of the best models to {model_paths_file}")
    
        return results

    def get_model_loss(self, model_path: str) -> float:
        """Extract validation loss from saved model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        return checkpoint['val_loss']
    

    def prepare_data_split(self, df, seed=1):
    
        db_lst = df.db.unique()
        if len(db_lst)>1: # Same split than individual datasets
            df_train = pd.DataFrame()
            df_valid = pd.DataFrame()
            for x in db_lst:
                idx = (df.db==x)
                tmp_train, tmp_valid = train_test_split(df.loc[idx,:], 
                                                    test_size = 0.2, 
                                                    shuffle = True, random_state = seed)
                df_train = pd.concat([df_train, tmp_train], ignore_index=True)
                df_valid = pd.concat([df_train, tmp_train], ignore_index=True)
        else:    
            df_train, df_valid = train_test_split(df, 
                                                test_size = 0.2, 
                                                shuffle = True, random_state = seed)    

        # Convert to PyTorch Geometric dataset
        train_dataset = create_dataset_from_dataframe(df_train)
        val_dataset = create_dataset_from_dataframe(df_valid)
        
        return train_dataset, val_dataset    

    def predict_with_ensemble(self, test_df):
        test_dataset = create_dataset_from_dataframe(test_df, result=False)
        test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
        
        all_predictions = []
        model_losses = []
        
        # Collect predictions and losses from all models
        for model_path in self.models:
            model = EdgeVGAE(1, 7, self.config.hidden_dim, 
                           self.config.latent_dim, 
                           self.config.num_classes).to(self.device)
            model_data = torch.load(model_path)
            model.load_state_dict(model_data['model_state_dict'])
            val_loss = model_data['val_loss']
            
            predictions = predict(model, self.device, test_loader)
            all_predictions.append(predictions)
            model_losses.append(val_loss)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        model_losses = np.array(model_losses)
        
        # Calculate weights using softmax of negative losses
        # Using negative losses because smaller loss should mean larger weight
        weights = np.exp(-model_losses)
        weights = weights / np.sum(weights)
        
        logging.info("Model weights for ensemble:")
        for model_path, weight, loss in zip(self.models, weights, model_losses):
            logging.info(f"Model: {os.path.basename(model_path)}")
            logging.info(f"- Loss: {loss:.4f}")
            logging.info(f"- Weight: {weight:.4f}")
        
        # Initialize array to store weighted votes for each class
        num_samples = all_predictions.shape[1]
        num_classes = self.config.num_classes
        weighted_votes = np.zeros((num_samples, num_classes))
        
        # Calculate weighted votes for each class
        for i, predictions in enumerate(all_predictions):
            for sample_idx in range(num_samples):
                pred_class = predictions[sample_idx]
                weighted_votes[sample_idx, pred_class] += weights[i]
        
        # Get the class with maximum weighted votes
        ensemble_predictions = np.argmax(weighted_votes, axis=1)
        
        # Calculate confidence scores
        confidence_scores = np.max(weighted_votes, axis=1)
        
        # Log some statistics about the ensemble predictions
        logging.info("\nEnsemble prediction statistics:")
        unique_preds, pred_counts = np.unique(ensemble_predictions, return_counts=True)
        for pred_class, count in zip(unique_preds, pred_counts):
            logging.info(f"Class {pred_class}: {count} samples")
        logging.info(f"Average confidence: {np.mean(confidence_scores):.4f}")
        logging.info(f"Min confidence: {np.min(confidence_scores):.4f}")
        logging.info(f"Max confidence: {np.max(confidence_scores):.4f}")
        
        return ensemble_predictions, confidence_scores
    
    def predict_with_ensemble_score(self, test_df):
        test_dataset = create_dataset_from_dataframe(test_df, result=False)
        test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
        
        all_predictions = []
        model_scores = []
        
        # Collect predictions and losses from all models
        for model_path in self.models:
            model = EdgeVGAE(1, 7, self.config.hidden_dim, 
                           self.config.latent_dim, 
                           self.config.num_classes).to(self.device)
            model_data = torch.load(model_path)
            model.load_state_dict(model_data['model_state_dict'])
            val_score = model_data['val_f1']
            
            predictions = predict(model, self.device, test_loader)
            all_predictions.append(predictions)
            model_scores.append(val_score)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        model_scores = np.array(model_scores)
        
        # Calculate weights using softmax of negative losses
        # Using negative losses because smaller loss should mean larger weight
        weights = np.exp(model_scores)
        weights = weights / np.sum(weights)
        
        logging.info("Model weights for ensemble:")
        for model_path, weight, loss in zip(self.models, weights, model_scores):
            logging.info(f"Model: {os.path.basename(model_path)}")
            logging.info(f"- Loss: {loss:.4f}")
            logging.info(f"- Weight: {weight:.4f}")
        
        # Initialize array to store weighted votes for each class
        num_samples = all_predictions.shape[1]
        num_classes = self.config.num_classes
        weighted_votes = np.zeros((num_samples, num_classes))
        
        # Calculate weighted votes for each class
        for i, predictions in enumerate(all_predictions):
            for sample_idx in range(num_samples):
                pred_class = predictions[sample_idx]
                weighted_votes[sample_idx, pred_class] += weights[i]
        
        # Get the class with maximum weighted votes
        ensemble_predictions = np.argmax(weighted_votes, axis=1)
        
        # Calculate confidence scores
        confidence_scores = np.max(weighted_votes, axis=1)
        
        # Log some statistics about the ensemble predictions
        logging.info("\nEnsemble prediction statistics:")
        unique_preds, pred_counts = np.unique(ensemble_predictions, return_counts=True)
        for pred_class, count in zip(unique_preds, pred_counts):
            logging.info(f"Class {pred_class}: {count} samples")
        logging.info(f"Average confidence: {np.mean(confidence_scores):.4f}")
        logging.info(f"Min confidence: {np.min(confidence_scores):.4f}")
        logging.info(f"Max confidence: {np.max(confidence_scores):.4f}")
        
        return ensemble_predictions, confidence_scores
    
    def predict_with_threshold(self, test_df, confidence_threshold=0.5):
        """
        Make predictions with a confidence threshold. Predictions below the threshold
        will be marked as -1 (uncertain).
        """
        predictions, confidences = self.predict_with_ensemble(test_df)
        
        # Mark low-confidence predictions as uncertain (-1)
        predictions[confidences < confidence_threshold] = -1
        
        logging.info(f"\nPredictions with confidence threshold {confidence_threshold}:")
        logging.info(f"Total predictions: {len(predictions)}")
        logging.info(f"Confident predictions: {np.sum(predictions != -1)}")
        logging.info(f"Uncertain predictions: {np.sum(predictions == -1)}")
        
        return predictions



def predict(model, device, loader):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)  # Move data to device if needed}

            z, mu, logvar, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch, eps=0.0)

            pred = class_logits.argmax(dim=1)  # Predicted class
            y_pred.extend(pred.tolist())
    
    return y_pred
