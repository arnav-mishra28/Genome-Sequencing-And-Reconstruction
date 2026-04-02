"""
DNA-BERT: Transformer Model for Genomic Sequence Analysis
========================================================

BERT-style transformer model for masked DNA prediction and sequence analysis.
Uses k-mer tokenization and pre-training on large genomic datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class KmerTokenizer:
    def __init__(self, k=6):
        """K-mer tokenizer for DNA sequences"""
        self.k = k
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build k-mer vocabulary"""
        bases = ['A', 'T', 'G', 'C']
        
        # Generate all possible k-mers
        def generate_kmers(length):
            if length == 1:
                return bases
            
            smaller_kmers = generate_kmers(length - 1)
            kmers = []
            for base in bases:
                for kmer in smaller_kmers:
                    kmers.append(base + kmer)
            return kmers
        
        # Special tokens
        special_tokens = ['[PAD]', '[MASK]', '[CLS]', '[SEP]', '[UNK]']
        
        # All k-mers
        kmers = generate_kmers(self.k)
        
        # Build vocabulary
        all_tokens = special_tokens + kmers
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        self.pad_token_id = self.vocab['[PAD]']
        self.mask_token_id = self.vocab['[MASK]']
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']
        self.unk_token_id = self.vocab['[UNK]']
    
    def tokenize(self, sequence, max_length=512):
        """Convert DNA sequence to k-mer tokens"""
        sequence = sequence.upper()
        
        # Extract k-mers
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if 'N' not in kmer:
                kmers.append(kmer)
            else:
                kmers.append('[UNK]')
        
        # Convert to token IDs
        token_ids = [self.cls_token_id]
        for kmer in kmers[:max_length-2]:  # Leave space for CLS and SEP
            token_ids.append(self.vocab.get(kmer, self.unk_token_id))
        token_ids.append(self.sep_token_id)
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token_id)
        
        return token_ids[:max_length]
    
    def detokenize(self, token_ids):
        """Convert token IDs back to DNA sequence"""
        tokens = [self.reverse_vocab[idx] for idx in token_ids 
                 if idx not in [self.pad_token_id, self.cls_token_id, self.sep_token_id]]
        
        # Reconstruct sequence from overlapping k-mers
        if not tokens:
            return ""
        
        sequence = tokens[0]
        for token in tokens[1:]:
            if token != '[UNK]' and token != '[MASK]':
                # Add the last base of the k-mer
                sequence += token[-1]
        
        return sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DNABERTModel(pl.LightningModule):
    def __init__(self, vocab_size=4**6 + 5, d_model=768, nhead=12, num_layers=6, 
                 dim_feedforward=3072, dropout=0.1, max_seq_len=512, learning_rate=5e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.masked_lm_head = nn.Linear(d_model, vocab_size)
        self.sequence_classification_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Healthy vs Diseased
        )
        
        # Mutation detection head
        self.mutation_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.learning_rate = learning_rate
    
    def forward(self, input_ids, attention_mask=None, task='mlm'):
        # Embeddings
        embedded = self.token_embedding(input_ids)
        embedded = self.positional_encoding(embedded)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Convert to transformer format (True = masked)
        src_key_padding_mask = ~attention_mask.bool()
        
        # Transformer
        encoded = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        
        if task == 'mlm':
            # Masked language modeling
            return self.masked_lm_head(encoded)
        elif task == 'classification':
            # Sequence classification (use CLS token)
            cls_representation = encoded[:, 0]  # CLS token
            return self.sequence_classification_head(cls_representation)
        elif task == 'mutation_detection':
            # Per-position mutation probability
            return self.mutation_head(encoded)
        
        return encoded
    
    def training_step(self, batch, batch_idx):
        if 'masked_input' in batch:
            # Masked language modeling
            predictions = self(batch['masked_input'], task='mlm')
            loss = F.cross_entropy(
                predictions.reshape(-1, predictions.size(-1)),
                batch['labels'].reshape(-1),
                ignore_index=-100
            )
            self.log('train_mlm_loss', loss, prog_bar=True)
            
        elif 'sequence_labels' in batch:
            # Sequence classification
            predictions = self(batch['input_ids'], task='classification')
            loss = F.cross_entropy(predictions, batch['sequence_labels'])
            self.log('train_cls_loss', loss, prog_bar=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        if 'masked_input' in batch:
            predictions = self(batch['masked_input'], task='mlm')
            loss = F.cross_entropy(
                predictions.reshape(-1, predictions.size(-1)),
                batch['labels'].reshape(-1),
                ignore_index=-100
            )
            
            # Calculate accuracy for masked positions only
            masked_positions = (batch['labels'] != -100)
            if masked_positions.any():
                predicted_tokens = torch.argmax(predictions, dim=-1)
                accuracy = (predicted_tokens == batch['labels'])[masked_positions].float().mean()
                self.log('val_mlm_accuracy', accuracy, prog_bar=True)
            
            self.log('val_mlm_loss', loss, prog_bar=True)
            
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

class MaskedDNADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=512, mask_prob=0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize
        token_ids = self.tokenizer.tokenize(sequence, self.max_length)
        
        # Create masked version for MLM
        masked_ids = token_ids.copy()
        labels = [-100] * len(token_ids)  # -100 is ignored in loss
        
        for i in range(1, len(token_ids) - 1):  # Skip CLS and SEP tokens
            if token_ids[i] != self.tokenizer.pad_token_id and random.random() < self.mask_prob:
                labels[i] = token_ids[i]  # Original token
                
                # 80% mask, 10% random, 10% keep
                rand = random.random()
                if rand < 0.8:
                    masked_ids[i] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    # Random token (excluding special tokens)
                    masked_ids[i] = random.randint(5, len(self.tokenizer.vocab) - 1)
                # 10% keep original (already set)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'masked_input': torch.tensor(masked_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1 if id != self.tokenizer.pad_token_id else 0 
                                          for id in token_ids], dtype=torch.long)
        }

class DNABERTForSequenceCompletion(DNABERTModel):
    """DNA-BERT specialized for sequence completion tasks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional layers for completion confidence
        self.completion_confidence = nn.Sequential(
            nn.Linear(self.hparams.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Multi-task output heads
        self.disease_predictor = nn.Sequential(
            nn.Linear(self.hparams.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(DISEASE_MUTATIONS)),
            nn.Sigmoid()
        )
    
    def predict_completion(self, incomplete_sequence, tokenizer):
        """Complete an incomplete DNA sequence"""
        self.eval()
        
        # Tokenize input
        token_ids = tokenizer.tokenize(incomplete_sequence)
        
        with torch.no_grad():
            # Get predictions
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            predictions = self(input_tensor, task='mlm')
            confidence = self.completion_confidence(self.transformer(self.token_embedding(input_tensor)))
            
            # Get probabilities
            probabilities = F.softmax(predictions, dim=-1)
            
            # Complete sequence
            completed_tokens = []
            confidence_scores = []
            
            for i, (token_id, probs, conf) in enumerate(zip(token_ids, probabilities[0], confidence[0])):
                if token_id == tokenizer.unk_token_id:
                    # Predict missing k-mer
                    predicted_token = torch.argmax(probs).item()
                    completed_tokens.append(predicted_token)
                    confidence_scores.append(conf.item())
                else:
                    completed_tokens.append(token_id)
                    confidence_scores.append(1.0)  # Known bases have full confidence
            
            # Convert back to sequence
            completed_sequence = tokenizer.detokenize(completed_tokens)
            
        return {
            'completed_sequence': completed_sequence,
            'confidence_scores': confidence_scores,
            'base_probabilities': probabilities[0].cpu().numpy()
        }
    
    def analyze_mutations(self, sequence, tokenizer):
        """Analyze potential mutations in a sequence"""
        self.eval()
        
        token_ids = tokenizer.tokenize(sequence)
        
        with torch.no_grad():
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            encoded = self.transformer(self.token_embedding(input_tensor))
            
            # Disease predictions
            disease_probs = self.disease_predictor(encoded[:, 0])  # Use CLS token
            
            # Mutation scores per position
            mutation_scores = self.mutation_head(encoded)
            
            # Identify high-risk positions
            high_risk_positions = []
            for i, score in enumerate(mutation_scores[0]):
                if score > 0.7:  # High mutation probability
                    high_risk_positions.append({
                        'position': i,
                        'mutation_score': score.item(),
                        'kmer': tokenizer.reverse_vocab.get(token_ids[i], 'UNK')
                    })
        
        return {
            'disease_probabilities': disease_probs[0].cpu().numpy(),
            'mutation_scores': mutation_scores[0].cpu().numpy(),
            'high_risk_positions': high_risk_positions
        }

def create_dna_bert_trainer(sequences, batch_size=16, max_epochs=50):
    """Create a trainer for DNA-BERT"""
    
    # Initialize tokenizer
    tokenizer = KmerTokenizer(k=6)
    
    # Create dataset
    dataset = MaskedDNADataset(sequences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = DNABERTForSequenceCompletion(
        vocab_size=len(tokenizer.vocab),
        d_model=768,
        nhead=12,
        num_layers=6
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        val_check_interval=0.25
    )
    
    return trainer, model, dataloader, tokenizer

if __name__ == "__main__":
    # Test the DNA-BERT model
    tokenizer = KmerTokenizer(k=6)
    model = DNABERTForSequenceCompletion(vocab_size=len(tokenizer.vocab))
    
    # Test tokenization
    test_sequence = "ATCGATCGATCG"
    tokens = tokenizer.tokenize(test_sequence)
    reconstructed = tokenizer.detokenize(tokens)
    
    print(f"Original: {test_sequence}")
    print(f"Tokens: {tokens[:10]}...")
    print(f"Reconstructed: {reconstructed}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
