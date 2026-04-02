#!/usr/bin/env python3
"""
Autoencoder Model for DNA Sequence Denoising
Removes noise and repairs damaged DNA sequences
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
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNADenoisingDataset(Dataset):
    """
    Dataset for DNA sequence denoising
    Creates noisy-clean pairs for training
    """
    def __init__(self, sequences, sequence_length=1000, noise_rate=0.2, damage_types=None):
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.noise_rate = noise_rate
        
        if damage_types is None:
            self.damage_types = ['substitution', 'deletion', 'insertion', 'deamination', 'missing']
        else:
            self.damage_types = damage_types
        
        # Base encoding
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        self.vocab_size = len(self.base_to_idx)
        
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """Create noisy-clean sequence pairs"""
        samples = []
        
        for seq in self.sequences:
            # Skip sequences that are too short
            if len(seq) < self.sequence_length:
                continue
            
            # Create multiple segments from each sequence
            for start in range(0, len(seq) - self.sequence_length + 1, self.sequence_length // 2):
                clean_segment = seq[start:start + self.sequence_length]
                
                # Skip segments with too many N's
                if clean_segment.count('N') / len(clean_segment) > 0.1:
                    continue
                
                # Create noisy version
                noisy_segment = self._add_noise(clean_segment)
                
                samples.append((noisy_segment, clean_segment))
        
        return samples
    
    def _add_noise(self, sequence):
        """
        Add various types of DNA damage/noise
        """
        damaged_seq = list(sequence)
        seq_length = len(damaged_seq)
        
        # Calculate number of damages based on noise rate
        num_damages = int(seq_length * self.noise_rate)
        
        for _ in range(num_damages):
            damage_type = random.choice(self.damage_types)
            position = random.randint(0, len(damaged_seq) - 1)
            
            if damage_type == 'substitution':
                # Random substitution
                original_base = damaged_seq[position]
                new_base = random.choice([b for b in 'ATGC' if b != original_base])
                damaged_seq[position] = new_base
                
            elif damage_type == 'deamination':
                # C->T deamination (common in ancient DNA)
                if damaged_seq[position] == 'C':
                    damaged_seq[position] = 'T'
                elif damaged_seq[position] == 'G':
                    damaged_seq[position] = 'A'
                    
            elif damage_type == 'deletion':
                # Delete base
                if len(damaged_seq) > 100:  # Keep minimum length
                    damaged_seq.pop(position)
                    
            elif damage_type == 'insertion':
                # Insert random base
                insert_base = random.choice('ATGC')
                damaged_seq.insert(position, insert_base)
                
            elif damage_type == 'missing':
                # Replace with N (missing data)
                damaged_seq[position] = 'N'
        
        # Ensure consistent length by padding or truncating
        damaged_str = ''.join(damaged_seq)
        if len(damaged_str) > len(sequence):
            damaged_str = damaged_str[:len(sequence)]
        elif len(damaged_str) < len(sequence):
            damaged_str += 'N' * (len(sequence) - len(damaged_str))
        
        return damaged_str
    
    def _encode_sequence(self, sequence):
        """Encode DNA sequence to numerical format"""
        return [self.base_to_idx.get(base, 4) for base in sequence.upper()]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        noisy_seq, clean_seq = self.samples[idx]
        
        noisy_encoded = torch.tensor(self._encode_sequence(noisy_seq), dtype=torch.long)
        clean_encoded = torch.tensor(self._encode_sequence(clean_seq), dtype=torch.long)
        
        return noisy_encoded, clean_encoded

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for DNA sequence denoising
    """
    def __init__(self, vocab_size=5, embedding_dim=64, hidden_dims=[128, 256, 512], 
                 sequence_length=1000, kernel_size=15):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        encoder_layers = []
        in_channels = embedding_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.1)
            ])
            in_channels = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck
        self.bottleneck_dim = hidden_dims[-1]
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.bottleneck_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            out_channels = hidden_dims_reversed[i+1] if i+1 < len(hidden_dims_reversed) else embedding_dim
            
            decoder_layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(hidden_dim, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv1d(embedding_dim, vocab_size, kernel_size=1),
            nn.AdaptiveAvgPool1d(sequence_length)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Encode
        encoded = self.encoder(embedded)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(encoded)
        
        # Reshape for decoder
        decoded_input = bottleneck_out.unsqueeze(-1).expand(-1, -1, encoded.size(-1))
        
        # Decode
        decoded = self.decoder(decoded_input)
        
        # Output
        output = self.output_layer(decoded)  # (batch_size, vocab_size, seq_len)
        output = output.transpose(1, 2)  # (batch_size, seq_len, vocab_size)
        
        return output

class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for DNA sequence denoising
    """
    def __init__(self, vocab_size=5, d_model=256, nhead=8, num_layers=6, 
                 sequence_length=1000, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU()
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Embedding and positional encoding
        embedded = self.embedding(x) * np.sqrt(self.d_model)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        embedded = embedded + pos_enc
        
        # Encoder
        memory = self.transformer_encoder(embedded)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(memory)
        
        # Decoder (using same input as target for denoising)
        decoded = self.transformer_decoder(embedded, bottleneck_out)
        
        # Output
        output = self.output_layer(decoded)
        
        return output

class DNAAutoencoder:
    """
    Main class for DNA sequence denoising using autoencoders
    """
    def __init__(self, model_dir="models/autoencoder", model_type="conv", device=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
        logger.info(f"Using device: {self.device}")
    
    def create_model(self, sequence_length=1000, **kwargs):
        """Create autoencoder model"""
        if self.model_type == "conv":
            self.model = ConvolutionalAutoencoder(
                vocab_size=5,
                sequence_length=sequence_length,
                **kwargs
            ).to(self.device)
        elif self.model_type == "transformer":
            self.model = TransformerAutoencoder(
                vocab_size=5,
                sequence_length=sequence_length,
                **kwargs
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} autoencoder with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train_model(self, train_sequences, val_sequences=None, epochs=30, batch_size=16, 
                   learning_rate=0.001, noise_rate=0.2):
        """Train the autoencoder"""
        logger.info("Preparing training data...")
        
        # Create datasets
        train_dataset = DNADenoisingDataset(train_sequences, noise_rate=noise_rate)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_sequences:
            val_dataset = DNADenoisingDataset(val_sequences, noise_rate=noise_rate)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=4)  # Ignore N's
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
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
            
            for batch_idx, (noisy_sequences, clean_sequences) in enumerate(train_loader):
                noisy_sequences = noisy_sequences.to(self.device)
                clean_sequences = clean_sequences.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(noisy_sequences)
                
                # Reshape for loss calculation
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = clean_sequences.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
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
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            
            scheduler.step()
            
            # Save checkpoint
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
            for noisy_sequences, clean_sequences in val_loader:
                noisy_sequences = noisy_sequences.to(self.device)
                clean_sequences = clean_sequences.to(self.device)
                
                outputs = self.model(noisy_sequences)
                
                # Calculate loss
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = clean_sequences.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                # Calculate accuracy (excluding N's)
                predictions = torch.argmax(outputs_flat, dim=1)
                
                non_n_mask = targets_flat != 4
                if non_n_mask.sum() > 0:
                    all_predictions.extend(predictions[non_n_mask].cpu().numpy())
                    all_targets.extend(targets_flat[non_n_mask].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
        
        return avg_val_loss, accuracy
    
    def denoise_sequence(self, noisy_sequence, confidence_threshold=0.5):
        """
        Denoise a damaged DNA sequence
        """
        self.model.eval()
        
        # Encode sequence
        encoded_seq = [self.base_to_idx.get(base.upper(), 4) for base in noisy_sequence]
        input_tensor = torch.tensor(encoded_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get predictions
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=-1)
            
            # Get most likely sequence
            predicted_indices = torch.argmax(probabilities, dim=-1)
            
            # Calculate confidence scores
            max_probs = torch.max(probabilities, dim=-1)[0]
            confidence_scores = max_probs.cpu().numpy()[0]
            
            # Convert back to sequence
            denoised_sequence = ""
            for i, (pred_idx, conf) in enumerate(zip(predicted_indices[0], confidence_scores)):
                if conf > confidence_threshold:
                    denoised_sequence += self.idx_to_base[pred_idx.item()]
                else:
                    # Keep original if confidence is low
                    original_char = noisy_sequence[i] if i < len(noisy_sequence) else 'N'
                    denoised_sequence += original_char
        
        return denoised_sequence, confidence_scores
    
    def batch_denoise(self, sequences, batch_size=32):
        """
        Denoise multiple sequences in batches
        """
        self.model.eval()
        denoised_sequences = []
        confidence_scores = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Find maximum length in batch
            max_length = max(len(seq) for seq in batch_sequences)
            
            # Encode and pad sequences
            batch_tensors = []
            for seq in batch_sequences:
                encoded = [self.base_to_idx.get(base.upper(), 4) for base in seq]
                # Pad to max length
                encoded.extend([4] * (max_length - len(encoded)))
                batch_tensors.append(encoded)
            
            input_tensor = torch.tensor(batch_tensors, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=-1)
                predicted_indices = torch.argmax(probabilities, dim=-1)
                max_probs = torch.max(probabilities, dim=-1)[0]
                
                # Convert back to sequences
                for j, (pred_seq, conf_seq, original_len) in enumerate(
                    zip(predicted_indices, max_probs, [len(seq) for seq in batch_sequences])
                ):
                    denoised_seq = ""
                    conf_scores = []
                    
                    for k in range(original_len):
                        denoised_seq += self.idx_to_base[pred_seq[k].item()]
                        conf_scores.append(conf_seq[k].item())
                    
                    denoised_sequences.append(denoised_seq)
                    confidence_scores.append(conf_scores)
        
        return denoised_sequences, confidence_scores
    
    def evaluate_denoising_quality(self, test_sequences, noise_rate=0.2):
        """
        Evaluate denoising performance on test sequences
        """
        # Create test dataset
        test_dataset = DNADenoisingDataset(test_sequences, noise_rate=noise_rate)
        
        accuracies = []
        recoveries = []  # Percentage of damaged bases correctly recovered
        
        for noisy_seq, clean_seq in test_dataset.samples[:100]:  # Test on subset
            denoised_seq, confidences = self.denoise_sequence(noisy_seq)
            
            # Calculate accuracy
            clean_str = ''.join([self.idx_to_base[idx] for idx in clean_seq])
            accuracy = sum(1 for a, b in zip(denoised_seq, clean_str) if a == b) / len(clean_str)
            accuracies.append(accuracy)
            
            # Calculate recovery rate for damaged positions
            damaged_positions = [i for i, (n, c) in enumerate(zip(noisy_seq, clean_str)) if n != c]
            if damaged_positions:
                recovered = sum(1 for i in damaged_positions 
                              if i < len(denoised_seq) and denoised_seq[i] == clean_str[i])
                recovery_rate = recovered / len(damaged_positions)
                recoveries.append(recovery_rate)
        
        results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_recovery_rate': np.mean(recoveries) if recoveries else 0,
            'std_recovery_rate': np.std(recoveries) if recoveries else 0
        }
        
        logger.info(f"Denoising evaluation: Accuracy {results['mean_accuracy']:.3f}±{results['std_accuracy']:.3f}")
        logger.info(f"Recovery rate: {results['mean_recovery_rate']:.3f}±{results['std_recovery_rate']:.3f}")
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
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
            'model_type': self.model_type,
            'model_config': self.model.__dict__ if hasattr(self.model, '__dict__') else {}
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
        
        # Recreate model based on saved type
        self.model_type = checkpoint['model_type']
        
        # This is simplified - in practice you'd save more detailed config
        if self.model_type == "conv":
            self.model = ConvolutionalAutoencoder().to(self.device)
        elif self.model_type == "transformer":
            self.model = TransformerAutoencoder().to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return True

def main():
    """
    Example usage of DNA autoencoder
    """
    # Test both model types
    for model_type in ["conv", "transformer"]:
        logger.info(f"\n=== Testing {model_type} autoencoder ===")
        
        autoencoder = DNAAutoencoder(model_type=model_type)
        
        # Create example sequences
        example_sequences = [
            "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 3,
            "TTGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA" * 3
        ]
        
        # Create and train model
        if model_type == "conv":
            model = autoencoder.create_model(sequence_length=300, embedding_dim=32, hidden_dims=[64, 128])
        else:
            model = autoencoder.create_model(sequence_length=300, d_model=128, num_layers=3, nhead=4)
        
        logger.info("Training autoencoder...")
        history = autoencoder.train_model(example_sequences, epochs=3, batch_size=2, noise_rate=0.3)
        
        # Test denoising
        noisy_sequence = "ATGCGANCGATCGANNNGATCGATCGATCGATCGATNGATCGATCGATCGNNGATCGATCGATCGATCGATCGATCGATC"
        denoised_sequence, confidences = autoencoder.denoise_sequence(noisy_sequence)
        
        logger.info(f"Noisy sequence:   {noisy_sequence}")
        logger.info(f"Denoised sequence: {denoised_sequence}")
        logger.info(f"Average confidence: {np.mean(confidences):.3f}")

if __name__ == "__main__":
    main()
