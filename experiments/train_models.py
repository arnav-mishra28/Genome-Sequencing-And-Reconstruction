"""
Training Scripts for Genome Sequencing Models

This module provides training scripts for LSTM, Transformer, Autoencoder,
and GNN models, along with ensemble training capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import logging
import os
import json
from datetime import datetime
import pickle
from tqdm import tqdm

# Import our models
import sys
sys.path.append('/home/claude/Genome Sequencing And Reconstruction')
from models.lstm_model import DNASequencePredictor
from models.transformer_model import DNATransformer
from models.autoencoder_model import ConvolutionalDNAAutoencoder, TransformerDNAAutoencoder
from models.gnn_model import PhylogeneticGNN, PhylogeneticGraphBuilder
from models.ensemble_model import DNAEnsemble
from utils.sequence_utils import SequenceEncoder, ModelEvaluator, DataLogger
from data.ancient_genome_downloader import AncientGenomeDataLoader
from preprocessing.dna_preprocessor import DNAPreprocessor


class DNASequenceDataset(Dataset):
    """Dataset class for DNA sequence training"""
    
    def __init__(self, 
                 sequences: List[str],
                 sequence_length: int = 1000,
                 encoding_type: str = 'integer',
                 task_type: str = 'prediction'):
        
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.encoder = SequenceEncoder(encoding_type)
        
        # Prepare data based on task type
        self.data = self._prepare_data()
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare training data based on task type"""
        data = []
        
        for sequence in self.sequences:
            if len(sequence) < self.sequence_length:
                # Pad with N
                sequence = sequence + 'N' * (self.sequence_length - len(sequence))
            else:
                sequence = sequence[:self.sequence_length]
            
            if self.task_type == 'prediction':
                # For next-base prediction
                input_seq = sequence[:-1]
                target_seq = sequence[1:]
                
                input_encoded = self.encoder.encode_sequence(input_seq)
                target_encoded = self.encoder.encode_sequence(target_seq)
                
                data.append({
                    'input': torch.tensor(input_encoded, dtype=torch.long),
                    'target': torch.tensor(target_encoded, dtype=torch.long)
                })
                
            elif self.task_type == 'masked_lm':
                # For masked language modeling
                encoded = self.encoder.encode_sequence(sequence)
                
                # Create random mask
                mask_prob = 0.15
                mask = np.random.random(len(encoded)) < mask_prob
                
                input_seq = encoded.copy()
                input_seq[mask] = 4  # Mask with N token
                
                data.append({
                    'input': torch.tensor(input_seq, dtype=torch.long),
                    'target': torch.tensor(encoded, dtype=torch.long),
                    'mask': torch.tensor(mask, dtype=torch.bool)
                })
                
            elif self.task_type == 'denoising':
                # For autoencoder denoising
                clean_encoded = self.encoder.encode_sequence(sequence)
                
                # Add noise
                noise_prob = 0.1
                noisy_encoded = clean_encoded.copy()
                noise_mask = np.random.random(len(noisy_encoded)) < noise_prob
                noisy_encoded[noise_mask] = np.random.randint(0, 5, size=np.sum(noise_mask))
                
                data.append({
                    'input': torch.tensor(noisy_encoded, dtype=torch.long),
                    'target': torch.tensor(clean_encoded, dtype=torch.long)
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class ModelTrainer:
    """Base trainer class for DNA sequence models"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 log_dir: str = 'training_logs'):
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model.to(self.device)
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def _setup_logger(self):
        """Setup logging"""
        logger = logging.getLogger(f'ModelTrainer_{id(self)}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def train(self,
              train_dataset: Dataset,
              val_dataset: Optional[Dataset] = None,
              epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              save_best: bool = True,
              early_stopping: bool = True,
              patience: int = 10) -> Dict:
        """Train the model"""
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=4)  # Ignore N tokens
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion) if val_loader else (0, 0)
            
            # Update learning rate
            if val_loader:
                scheduler.step(val_loss)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                f"{f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}' if val_loader else ''}"
            )
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            if val_loader:
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
            
            # Early stopping and save best model
            if val_loader and save_best:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_model(f'best_model_epoch_{epoch+1}.pth')
                else:
                    patience_counter += 1
                    
                if early_stopping and patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model
        self._save_model('final_model.pth')
        
        # Save training history
        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.training_history
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            outputs = self.model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (excluding N tokens)
            predictions = torch.argmax(outputs, dim=-1)
            mask = targets != 4  # Exclude N tokens
            if mask.sum() > 0:
                correct = (predictions == targets) & mask
                total_correct += correct.sum().item()
                total_samples += mask.sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        if not val_loader:
            return 0, 0
            
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=-1)
                mask = targets != 4
                if mask.sum() > 0:
                    correct = (predictions == targets) & mask
                    total_correct += correct.sum().item()
                    total_samples += mask.sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.log_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_accuracy'], label='Train Accuracy')
        if self.training_history['val_accuracy']:
            ax2.plot(self.training_history['val_accuracy'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'))
        plt.show()


class ExperimentRunner:
    """Run comprehensive experiments on all models"""
    
    def __init__(self, data_dir: str = '/home/claude/Genome Sequencing And Reconstruction/data'):
        self.data_dir = data_dir
        self.logger = DataLogger()
        self.results = {}
        
    def prepare_datasets(self, 
                        max_sequences: int = 1000,
                        sequence_length: int = 1000,
                        test_size: float = 0.2,
                        val_size: float = 0.1) -> Dict:
        """Prepare training, validation, and test datasets"""
        
        print("Loading and preparing datasets...")
        
        # Load ancient genome data
        data_loader = AncientGenomeDataLoader()
        preprocessor = DNAPreprocessor()
        
        # Load sequences from different species
        all_sequences = []
        
        # Add human genome data (simulated)
        human_sequences = data_loader.simulate_human_sequences(num_sequences=max_sequences//4)
        all_sequences.extend(human_sequences)
        
        # Add Neanderthal data (simulated)
        neanderthal_sequences = data_loader.simulate_ancient_dna('neanderthal', max_sequences//4)
        all_sequences.extend(neanderthal_sequences)
        
        # Add mammoth data (simulated) 
        mammoth_sequences = data_loader.simulate_ancient_dna('mammoth', max_sequences//4)
        all_sequences.extend(mammoth_sequences)
        
        # Add synthetic sequences
        synthetic_sequences = data_loader.generate_synthetic_sequences(max_sequences//4, sequence_length)
        all_sequences.extend(synthetic_sequences)
        
        print(f"Total sequences loaded: {len(all_sequences)}")
        
        # Clean and preprocess
        cleaned_sequences = []
        for seq in all_sequences:
            cleaned = preprocessor.clean_sequence(seq)
            if len(cleaned) >= sequence_length // 2:  # Keep sequences that are at least half the target length
                cleaned_sequences.append(cleaned)
        
        print(f"Sequences after cleaning: {len(cleaned_sequences)}")
        
        # Split datasets
        total = len(cleaned_sequences)
        test_split = int(total * test_size)
        val_split = int(total * val_size)
        train_split = total - test_split - val_split
        
        train_sequences = cleaned_sequences[:train_split]
        val_sequences = cleaned_sequences[train_split:train_split + val_split]
        test_sequences = cleaned_sequences[train_split + val_split:]
        
        # Create datasets for different tasks
        datasets = {
            'prediction': {
                'train': DNASequenceDataset(train_sequences, sequence_length, 'integer', 'prediction'),
                'val': DNASequenceDataset(val_sequences, sequence_length, 'integer', 'prediction'),
                'test': DNASequenceDataset(test_sequences, sequence_length, 'integer', 'prediction')
            },
            'masked_lm': {
                'train': DNASequenceDataset(train_sequences, sequence_length, 'integer', 'masked_lm'),
                'val': DNASequenceDataset(val_sequences, sequence_length, 'integer', 'masked_lm'),
                'test': DNASequenceDataset(test_sequences, sequence_length, 'integer', 'masked_lm')
            },
            'denoising': {
                'train': DNASequenceDataset(train_sequences, sequence_length, 'integer', 'denoising'),
                'val': DNASequenceDataset(val_sequences, sequence_length, 'integer', 'denoising'),
                'test': DNASequenceDataset(test_sequences, sequence_length, 'integer', 'denoising')
            }
        }
        
        print(f"Dataset sizes - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
        
        return datasets
    
    def run_lstm_experiment(self, datasets: Dict, config: Dict = None) -> Dict:
        """Run LSTM training experiment"""
        print("\n=== LSTM Experiment ===")
        
        default_config = {
            'vocab_size': 5,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'max_length': 1000,
            'epochs': 30,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        if config:
            default_config.update(config)
        
        # Initialize model
        model = DNASequencePredictor(
            vocab_size=default_config['vocab_size'],
            embedding_dim=default_config['embedding_dim'],
            hidden_dim=default_config['hidden_dim'],
            num_layers=default_config['num_layers'],
            max_length=default_config['max_length']
        )
        
        # Train model
        trainer = ModelTrainer(model, log_dir='training_logs/lstm')
        history = trainer.train(
            train_dataset=datasets['prediction']['train'],
            val_dataset=datasets['prediction']['val'],
            epochs=default_config['epochs'],
            batch_size=default_config['batch_size'],
            learning_rate=default_config['learning_rate']
        )
        
        # Plot training history
        trainer.plot_training_history()
        
        return {
            'model_type': 'LSTM',
            'config': default_config,
            'training_history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
        }
    
    def run_transformer_experiment(self, datasets: Dict, config: Dict = None) -> Dict:
        """Run Transformer training experiment"""
        print("\n=== Transformer Experiment ===")
        
        default_config = {
            'vocab_size': 5,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'max_length': 1000,
            'epochs': 25,
            'batch_size': 16,
            'learning_rate': 0.0001
        }
        
        if config:
            default_config.update(config)
        
        # Initialize model
        model = DNATransformer(
            vocab_size=default_config['vocab_size'],
            d_model=default_config['d_model'],
            nhead=default_config['nhead'],
            num_layers=default_config['num_layers'],
            max_length=default_config['max_length']
        )
        
        # Train model
        trainer = ModelTrainer(model, log_dir='training_logs/transformer')
        history = trainer.train(
            train_dataset=datasets['masked_lm']['train'],
            val_dataset=datasets['masked_lm']['val'],
            epochs=default_config['epochs'],
            batch_size=default_config['batch_size'],
            learning_rate=default_config['learning_rate']
        )
        
        trainer.plot_training_history()
        
        return {
            'model_type': 'Transformer',
            'config': default_config,
            'training_history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
        }
    
    def run_autoencoder_experiment(self, datasets: Dict, config: Dict = None) -> Dict:
        """Run Autoencoder training experiment"""
        print("\n=== Autoencoder Experiment ===")
        
        default_config = {
            'input_channels': 5,
            'sequence_length': 1000,
            'epochs': 40,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        if config:
            default_config.update(config)
        
        # Train convolutional autoencoder
        conv_model = ConvolutionalDNAAutoencoder(
            input_channels=default_config['input_channels'],
            sequence_length=default_config['sequence_length']
        )
        
        conv_trainer = ModelTrainer(conv_model, log_dir='training_logs/conv_autoencoder')
        conv_history = conv_trainer.train(
            train_dataset=datasets['denoising']['train'],
            val_dataset=datasets['denoising']['val'],
            epochs=default_config['epochs'],
            batch_size=default_config['batch_size'],
            learning_rate=default_config['learning_rate']
        )
        
        conv_trainer.plot_training_history()
        
        return {
            'model_type': 'Convolutional Autoencoder',
            'config': default_config,
            'training_history': conv_history,
            'final_train_loss': conv_history['train_loss'][-1],
            'final_val_loss': conv_history['val_loss'][-1] if conv_history['val_loss'] else None
        }
    
    def run_full_experiment_suite(self, 
                                max_sequences: int = 1000,
                                sequence_length: int = 1000) -> Dict:
        """Run complete experimental suite"""
        
        self.logger.start_experiment('full_genome_reconstruction_suite', {
            'max_sequences': max_sequences,
            'sequence_length': sequence_length,
            'timestamp': datetime.now().isoformat()
        })
        
        print("Starting comprehensive DNA reconstruction experiments...")
        
        # Prepare datasets
        datasets = self.prepare_datasets(max_sequences, sequence_length)
        
        # Run individual experiments
        results = {}
        
        try:
            results['lstm'] = self.run_lstm_experiment(datasets)
        except Exception as e:
            print(f"LSTM experiment failed: {e}")
            results['lstm'] = {'error': str(e)}
        
        try:
            results['transformer'] = self.run_transformer_experiment(datasets)
        except Exception as e:
            print(f"Transformer experiment failed: {e}")
            results['transformer'] = {'error': str(e)}
        
        try:
            results['autoencoder'] = self.run_autoencoder_experiment(datasets)
        except Exception as e:
            print(f"Autoencoder experiment failed: {e}")
            results['autoencoder'] = {'error': str(e)}
        
        # Summary
        summary = {
            'experiment_completed': datetime.now().isoformat(),
            'total_sequences': max_sequences,
            'sequence_length': sequence_length,
            'models_trained': len([k for k, v in results.items() if 'error' not in v]),
            'results': results
        }
        
        # Save results
        self.logger.log_results(summary)
        self.results = summary
        
        print("\n=== Experiment Suite Completed ===")
        print(f"Models successfully trained: {summary['models_trained']}")
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    
    print("Starting DNA Sequence Model Training...")
    
    # Run quick test with small dataset
    runner = ExperimentRunner()
    
    # Run a smaller test experiment
    test_results = runner.run_full_experiment_suite(
        max_sequences=100,  # Smaller for testing
        sequence_length=500
    )
    
    print("\nTraining completed! Results saved to logs.")
    
    # Display final summary
    if 'results' in test_results:
        for model_name, result in test_results['results'].items():
            if 'error' not in result:
                print(f"\n{model_name.upper()} Results:")
                print(f"  Final training loss: {result.get('final_train_loss', 'N/A'):.4f}")
                if result.get('final_val_loss'):
                    print(f"  Final validation loss: {result['final_val_loss']:.4f}")
            else:
                print(f"\n{model_name.upper()}: Failed with error: {result['error']}")
