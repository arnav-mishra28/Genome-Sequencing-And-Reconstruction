#!/usr/bin/env python3
"""
LSTM Model for DNA Sequence Prediction
Predicts next nucleotides in incomplete sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNASequenceDataset(Dataset):
    """
    Dataset class for DNA sequences
    """
    def __init__(self, sequences, sequence_length=100, stride=50, encoding='integer'):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.stride = stride
        self.encoding = encoding
        
        # Base encoding
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        self.vocab_size = len(self.base_to_idx)
        
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """Create input-output pairs for training"""
        samples = []
        
        for seq in self.sequences:
            # Skip sequences that are too short
            if len(seq) < self.sequence_length + 1:
                continue
                
            # Create sliding windows
            for i in range(0, len(seq) - self.sequence_length, self.stride):
                input_seq = seq[i:i + self.sequence_length]
                target_seq = seq[i + 1:i + self.sequence_length + 1]
                
                # Skip if too many N's
                if input_seq.count('N') / len(input_seq) > 0.3:
                    continue
                
                samples.append((input_seq, target_seq))
        
        return samples
    
    def _encode_sequence(self, sequence):
        """Encode DNA sequence to numerical format"""
        if self.encoding == 'integer':
            return [self.base_to_idx[base] for base in sequence]
        elif self.encoding == 'onehot':
            encoded = np.zeros((len(sequence), self.vocab_size))
            for i, base in enumerate(sequence):
                encoded[i, self.base_to_idx[base]] = 1
            return encoded
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        
        input_encoded = torch.tensor(self._encode_sequence(input_seq), dtype=torch.long)
        target_encoded = torch.tensor(self._encode_sequence(target_seq), dtype=torch.long)
        
        return input_encoded, target_encoded

class DNASequenceLSTM(nn.Module):
    """
    LSTM model for DNA sequence prediction
    """
    def __init__(self, vocab_size=5, embedding_dim=64, hidden_dim=256, num_layers=3, dropout=0.2):
        super(DNASequenceLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.use_attention = True
        
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Optional attention
        if self.use_attention:
            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attended_out  # Residual connection
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        output = self.output_layer(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class DNASequencePredictor:
    """
    Main class for training and using the LSTM DNA predictor
    """
    def __init__(self, model_dir="models/lstm", device=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
    def create_model(self, vocab_size=5, embedding_dim=64, hidden_dim=256, num_layers=3):
        """Create LSTM model"""
        self.model = DNASequenceLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        logger.info(f"Created LSTM model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train_model(self, train_sequences, val_sequences=None, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        logger.info("Preparing training data...")
        
        # Create datasets
        train_dataset = DNASequenceDataset(train_sequences, sequence_length=100, encoding='integer')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_sequences:
            val_dataset = DNASequenceDataset(val_sequences, sequence_length=100, encoding='integer')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=4)  # Ignore N's in loss calculation
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.model(inputs)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, self.model.vocab_size)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_sequences:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                scheduler.step(val_loss)
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model and training history
        self.save_model("final_model.pth")
        
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        with open(self.model_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(training_history, f)
        
        self.plot_training_history(training_history)
        
        logger.info("Training completed!")
        return training_history
    
    def _validate(self, val_loader, criterion):
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs, _ = self.model(inputs)
                
                # Reshape for loss calculation
                outputs_flat = outputs.view(-1, self.model.vocab_size)
                targets_flat = targets.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                # Calculate accuracy (excluding N's)
                predictions = torch.argmax(outputs_flat, dim=1)
                
                # Only consider non-N positions for accuracy
                non_n_mask = targets_flat != 4
                if non_n_mask.sum() > 0:
                    all_predictions.extend(predictions[non_n_mask].cpu().numpy())
                    all_targets.extend(targets_flat[non_n_mask].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
        
        return avg_val_loss, accuracy
    
    def predict_sequence(self, partial_sequence, length_to_predict=100, temperature=1.0):
        """
        Predict next nucleotides in sequence
        """
        self.model.eval()
        
        # Encode input sequence
        input_seq = [self.base_to_idx[base] for base in partial_sequence.upper()]
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        
        predicted_sequence = list(partial_sequence.upper())
        hidden = self.model.init_hidden(1, self.device)
        
        with torch.no_grad():
            # Process input sequence first
            _, hidden = self.model(input_tensor, hidden)
            
            # Generate new sequence
            current_input = input_tensor[:, -1:]  # Last nucleotide
            
            for _ in range(length_to_predict):
                outputs, hidden = self.model(current_input, hidden)
                
                # Apply temperature scaling
                outputs = outputs / temperature
                probabilities = torch.softmax(outputs[:, -1, :], dim=-1)
                
                # Sample next nucleotide
                next_base_idx = torch.multinomial(probabilities, 1).item()
                
                # Skip N's in generation
                if next_base_idx == 4:  # N
                    continue
                
                next_base = self.idx_to_base[next_base_idx]
                predicted_sequence.append(next_base)
                
                # Update input for next prediction
                current_input = torch.tensor([[next_base_idx]], dtype=torch.long).to(self.device)
        
        return ''.join(predicted_sequence)
    
    def calculate_confidence_score(self, sequence, window_size=50):
        """
        Calculate confidence score for each position in sequence
        """
        self.model.eval()
        confidence_scores = []
        
        with torch.no_grad():
            for i in range(len(sequence) - window_size + 1):
                window = sequence[i:i + window_size]
                
                # Encode window
                input_seq = [self.base_to_idx[base] for base in window]
                input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get predictions
                outputs, _ = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=-1)
                
                # Calculate entropy-based confidence
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
                max_entropy = np.log(self.model.vocab_size)
                confidence = 1 - entropy.mean().item() / max_entropy
                
                confidence_scores.append(confidence)
        
        return confidence_scores
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_losses'], label='Train Loss')
        if history['val_losses']:
            axes[0].plot(history['val_losses'], label='Validation Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy plot
        if history['val_accuracies']:
            axes[1].plot(history['val_accuracies'], label='Validation Accuracy')
            axes[1].set_title('Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
        
        # Learning rate (if available)
        axes[2].text(0.5, 0.5, 'Additional metrics\ncan be plotted here', 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Additional Metrics')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training history plot saved")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers
        }
        
        filepath = self.model_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model with saved parameters
        self.model = DNASequenceLSTM(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return True

def main():
    """
    Example usage of the LSTM DNA predictor
    """
    predictor = DNASequencePredictor()
    
    # Create example sequences for demonstration
    example_sequences = [
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
        "TTGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA",
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGT"
    ]
    
    # Create and train model
    model = predictor.create_model()
    
    # For demonstration with small data
    logger.info("Training LSTM model...")
    history = predictor.train_model(example_sequences, epochs=5, batch_size=2)
    
    # Test prediction
    partial_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGNNNNNNNNNN"
    predicted = predictor.predict_sequence(partial_sequence, length_to_predict=20)
    
    logger.info(f"Input sequence: {partial_sequence}")
    logger.info(f"Predicted sequence: {predicted}")
    
    # Calculate confidence
    confidence = predictor.calculate_confidence_score(predicted)
    logger.info(f"Average confidence: {np.mean(confidence):.3f}")

if __name__ == "__main__":
    main()
