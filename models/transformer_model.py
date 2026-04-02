#!/usr/bin/env python3
"""
Transformer Model for DNA Sequence Analysis
DNABERT-style masked language modeling for DNA
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
from transformers import AutoTokenizer, AutoModel, BertConfig, BertForMaskedLM
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KmerTokenizer:
    """
    K-mer tokenizer for DNA sequences
    Similar to DNABERT tokenization
    """
    def __init__(self, k=6):
        self.k = k
        self.vocab = self._create_kmer_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.unk_token = '[UNK]'
        
    def _create_kmer_vocab(self):
        """Create all possible k-mers"""
        bases = ['A', 'T', 'G', 'C']
        kmers = []
        
        def generate_kmers(current, remaining):
            if remaining == 0:
                kmers.append(current)
                return
            for base in bases:
                generate_kmers(current + base, remaining - 1)
        
        generate_kmers('', self.k)
        
        # Add special tokens
        special_tokens = ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]']
        vocab = special_tokens + kmers
        
        return vocab
    
    def sequence_to_kmers(self, sequence, stride=1):
        """Convert DNA sequence to k-mers"""
        sequence = sequence.upper()
        kmers = []
        
        for i in range(0, len(sequence) - self.k + 1, stride):
            kmer = sequence[i:i + self.k]
            if 'N' not in kmer:  # Skip k-mers with unknown bases
                kmers.append(kmer)
            else:
                kmers.append('[UNK]')
        
        return kmers
    
    def encode(self, sequence, max_length=512, add_special_tokens=True):
        """Encode DNA sequence to token IDs"""
        kmers = self.sequence_to_kmers(sequence)
        
        if add_special_tokens:
            kmers = ['[CLS]'] + kmers + ['[SEP]']
        
        # Truncate or pad
        if len(kmers) > max_length:
            kmers = kmers[:max_length]
        else:
            kmers.extend(['[PAD]'] * (max_length - len(kmers)))
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(kmer, self.token_to_id['[UNK]']) for kmer in kmers]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if token != '[PAD]' else 0 for token in kmers], dtype=torch.long)
        }
    
    def decode(self, token_ids):
        """Decode token IDs back to k-mers"""
        return [self.id_to_token.get(token_id, '[UNK]') for token_id in token_ids]
    
    def kmers_to_sequence(self, kmers, stride=1):
        """Convert k-mers back to DNA sequence"""
        if not kmers:
            return ""
        
        # Remove special tokens
        kmers = [kmer for kmer in kmers if kmer not in ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]']]
        
        if not kmers:
            return ""
        
        sequence = kmers[0]
        for i in range(1, len(kmers)):
            if kmers[i] != '[UNK]':
                # Add only the last nucleotide of each k-mer (assuming stride=1)
                sequence += kmers[i][-stride:]
        
        return sequence

class DNAMaskedLMDataset(Dataset):
    """
    Dataset for DNA masked language modeling
    """
    def __init__(self, sequences, tokenizer, max_length=512, mask_prob=0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        encoding = self.tokenizer.encode(sequence, max_length=self.max_length)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        
        # Apply masking
        masked_input_ids = self._apply_masking(input_ids.clone())
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _apply_masking(self, input_ids):
        """Apply BERT-style masking"""
        # Don't mask special tokens
        special_token_ids = [
            self.tokenizer.token_to_id['[PAD]'],
            self.tokenizer.token_to_id['[CLS]'],
            self.tokenizer.token_to_id['[SEP]']
        ]
        
        for i in range(len(input_ids)):
            if input_ids[i].item() in special_token_ids:
                continue
                
            if random.random() < self.mask_prob:
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    input_ids[i] = self.tokenizer.token_to_id['[MASK]']
                # 10% of the time, replace with random token
                elif random.random() < 0.5:
                    random_token_id = random.randint(5, len(self.tokenizer.vocab) - 1)  # Skip special tokens
                    input_ids[i] = random_token_id
                # 10% of the time, keep original
        
        return input_ids

class DNABERTModel(nn.Module):
    """
    BERT-style transformer for DNA sequences
    """
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, 
                 intermediate_size=3072, max_position_embeddings=512, dropout=0.1):
        super(DNABERTModel, self).__init__()
        
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        # Initialize BERT model
        self.bert = BertForMaskedLM(self.config)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

class DNATransformerPredictor:
    """
    Main class for DNA Transformer model
    """
    def __init__(self, model_dir="models/transformer", k=6, device=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = KmerTokenizer(k=k)
        self.model = None
        
    def create_model(self, hidden_size=256, num_layers=6, num_heads=8):
        """Create DNA BERT model"""
        vocab_size = len(self.tokenizer.vocab)
        
        self.model = DNABERTModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512
        ).to(self.device)
        
        logger.info(f"Created DNA BERT model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train_model(self, train_sequences, val_sequences=None, epochs=20, batch_size=16, learning_rate=5e-5):
        """Train the transformer model"""
        logger.info("Preparing training data...")
        
        # Create datasets
        train_dataset = DNAMaskedLMDataset(train_sequences, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_sequences:
            val_dataset = DNAMaskedLMDataset(val_sequences, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        num_training_steps = len(train_loader) * epochs
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)
        
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
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_sequences:
                val_loss, val_accuracy = self._validate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
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
        
        logger.info("Training completed!")
        return training_history
    
    def _validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                
                # Calculate accuracy for masked tokens
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Only calculate accuracy for actually masked positions
                mask_token_id = self.tokenizer.token_to_id['[MASK]']
                masked_positions = input_ids == mask_token_id
                
                if masked_positions.sum() > 0:
                    masked_predictions = predictions[masked_positions]
                    masked_targets = labels[masked_positions]
                    
                    all_predictions.extend(masked_predictions.cpu().numpy())
                    all_targets.extend(masked_targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) if all_predictions else 0.0
        
        return avg_val_loss, accuracy
    
    def predict_masked_sequence(self, sequence_with_masks, top_k=5):
        """
        Predict masked nucleotides in sequence
        """
        self.model.eval()
        
        # Find mask positions
        mask_positions = [i for i, char in enumerate(sequence_with_masks) if char == 'N']
        
        # Replace N's with [MASK] tokens for k-mer tokenization
        masked_sequence = sequence_with_masks.replace('N', 'N' * self.tokenizer.k)
        
        # Tokenize
        encoding = self.tokenizer.encode(masked_sequence)
        input_ids = encoding['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits
            
            # Get top-k predictions for each position
            top_k_predictions = torch.topk(predictions, k=top_k, dim=-1)
            
            predicted_tokens = []
            confidence_scores = []
            
            mask_token_id = self.tokenizer.token_to_id['[MASK]']
            for i, token_id in enumerate(input_ids[0]):
                if token_id == mask_token_id:
                    top_tokens = top_k_predictions.indices[0, i]
                    top_probs = torch.softmax(predictions[0, i], dim=-1)[top_tokens]
                    
                    predicted_tokens.append([self.tokenizer.id_to_token[tid.item()] for tid in top_tokens])
                    confidence_scores.append(top_probs.cpu().numpy())
        
        return predicted_tokens, confidence_scores
    
    def fill_sequence_gaps(self, incomplete_sequence, confidence_threshold=0.5):
        """
        Fill gaps in incomplete DNA sequence
        """
        # Replace gaps with mask tokens
        sequence_with_masks = ""
        for char in incomplete_sequence.upper():
            if char in 'ATGC':
                sequence_with_masks += char
            else:
                sequence_with_masks += 'N'
        
        # Get predictions
        predicted_tokens, confidences = self.predict_masked_sequence(sequence_with_masks)
        
        # Reconstruct sequence
        completed_sequence = list(incomplete_sequence.upper())
        prediction_confidences = []
        
        mask_idx = 0
        for i, char in enumerate(completed_sequence):
            if char not in 'ATGC':
                if mask_idx < len(predicted_tokens):
                    # Use top prediction if confidence is high enough
                    top_token = predicted_tokens[mask_idx][0]
                    top_confidence = confidences[mask_idx][0]
                    
                    if top_confidence > confidence_threshold and top_token != '[MASK]':
                        # Convert k-mer back to nucleotide (simplified)
                        if len(top_token) == self.tokenizer.k:
                            completed_sequence[i] = top_token[0]  # Use first nucleotide
                        
                    prediction_confidences.append(top_confidence)
                    mask_idx += 1
                else:
                    prediction_confidences.append(0.0)
        
        return ''.join(completed_sequence), prediction_confidences
    
    def calculate_sequence_quality(self, sequence, window_size=100):
        """
        Calculate quality score for DNA sequence
        """
        quality_scores = []
        
        for i in range(0, len(sequence) - window_size + 1, window_size // 2):
            window = sequence[i:i + window_size]
            
            # Create random mask pattern
            masked_window = ""
            original_positions = []
            
            for j, char in enumerate(window):
                if random.random() < 0.15 and char in 'ATGC':  # 15% mask rate
                    masked_window += 'N'
                    original_positions.append((j, char))
                else:
                    masked_window += char
            
            if original_positions:
                # Get predictions for masked positions
                predicted_tokens, confidences = self.predict_masked_sequence(masked_window)
                
                # Calculate accuracy
                correct_predictions = 0
                total_predictions = len(original_positions)
                
                for k, (pos, original_char) in enumerate(original_positions):
                    if k < len(predicted_tokens):
                        top_token = predicted_tokens[k][0]
                        if len(top_token) == self.tokenizer.k and top_token[0] == original_char:
                            correct_predictions += 1
                
                quality = correct_predictions / total_predictions if total_predictions > 0 else 1.0
                quality_scores.append(quality)
        
        return quality_scores
    
    def save_model(self, filename):
        """Save model checkpoint"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab': self.tokenizer.vocab,
            'tokenizer_k': self.tokenizer.k,
            'config': self.model.config.__dict__
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
        
        # Restore tokenizer
        self.tokenizer.vocab = checkpoint['tokenizer_vocab']
        self.tokenizer.k = checkpoint['tokenizer_k']
        self.tokenizer.token_to_id = {token: idx for idx, token in enumerate(self.tokenizer.vocab)}
        self.tokenizer.id_to_token = {idx: token for token, idx in self.tokenizer.token_to_id.items()}
        
        # Create model
        config = checkpoint['config']
        self.model = DNABERTModel(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=config['hidden_size'],
            num_layers=config['num_hidden_layers'],
            num_heads=config['num_attention_heads']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return True

def main():
    """
    Example usage of DNA Transformer model
    """
    predictor = DNATransformerPredictor()
    
    # Create example sequences
    example_sequences = [
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 3,
        "TTGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA" * 3,
        "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGT" * 3
    ]
    
    # Create and train model
    model = predictor.create_model(hidden_size=128, num_layers=4, num_heads=4)  # Smaller for demo
    
    logger.info("Training DNA Transformer model...")
    history = predictor.train_model(example_sequences, epochs=3, batch_size=2)
    
    # Test prediction
    incomplete_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGNNNNNNNNNNGATCGATCGATCGATC"
    completed_sequence, confidences = predictor.fill_sequence_gaps(incomplete_sequence)
    
    logger.info(f"Input sequence: {incomplete_sequence}")
    logger.info(f"Completed sequence: {completed_sequence}")
    logger.info(f"Average confidence: {np.mean(confidences):.3f}")

if __name__ == "__main__":
    main()
