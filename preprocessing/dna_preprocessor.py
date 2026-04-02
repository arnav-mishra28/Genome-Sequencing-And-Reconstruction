#!/usr/bin/env python3
"""
DNA Sequence Preprocessing and Alignment
Handles sequence cleaning, alignment, and numerical conversion
"""

import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
import subprocess
import logging
from collections import Counter
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNAPreprocessor:
    def __init__(self, data_dir="data", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # DNA encoding schemes
        self.base_to_num = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.num_to_base = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
        # One-hot encoding
        self.base_to_onehot = {
            'A': [1, 0, 0, 0, 0],
            'T': [0, 1, 0, 0, 0],
            'G': [0, 0, 1, 0, 0],
            'C': [0, 0, 0, 1, 0],
            'N': [0, 0, 0, 0, 1]
        }
        
        # K-mer tokenization for transformer models
        self.kmer_size = 6  # 6-mers like DNABERT
        
    def clean_sequence(self, sequence, min_length=100):
        """
        Clean DNA sequence:
        - Remove invalid characters
        - Filter short sequences
        - Handle ambiguous bases
        """
        # Convert to uppercase
        cleaned = str(sequence).upper()
        
        # Remove non-DNA characters
        valid_chars = set('ATGCN')
        cleaned = ''.join(c if c in valid_chars else 'N' for c in cleaned)
        
        # Filter by length
        if len(cleaned) < min_length:
            return None
            
        return cleaned
    
    def segment_sequence(self, sequence, segment_length=10000, overlap=1000):
        """
        Segment long sequences into smaller overlapping chunks
        Necessary for deep learning models with memory constraints
        """
        segments = []
        seq_len = len(sequence)
        
        start = 0
        while start < seq_len:
            end = min(start + segment_length, seq_len)
            segment = sequence[start:end]
            
            if len(segment) >= segment_length // 2:  # Keep segments at least half the target length
                segments.append({
                    'sequence': segment,
                    'start': start,
                    'end': end,
                    'length': len(segment)
                })
            
            start += segment_length - overlap
            if end >= seq_len:
                break
                
        return segments
    
    def create_kmers(self, sequence, k=6, stride=1):
        """
        Create k-mer tokens from DNA sequence
        Used for transformer-based models like DNABERT
        """
        kmers = []
        for i in range(0, len(sequence) - k + 1, stride):
            kmer = sequence[i:i+k]
            if 'N' not in kmer:  # Skip k-mers with unknown bases
                kmers.append(kmer)
        return kmers
    
    def sequence_to_numerical(self, sequence, encoding='integer'):
        """
        Convert DNA sequence to numerical representation
        encoding: 'integer', 'onehot', 'kmer'
        """
        if encoding == 'integer':
            return np.array([self.base_to_num[base] for base in sequence])
        
        elif encoding == 'onehot':
            return np.array([self.base_to_onehot[base] for base in sequence])
        
        elif encoding == 'kmer':
            kmers = self.create_kmers(sequence, self.kmer_size)
            # Create k-mer vocabulary (simplified - in practice use pre-trained vocab)
            return kmers
        
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    
    def create_kmer_vocabulary(self, sequences, k=6, min_freq=5):
        """
        Create k-mer vocabulary from sequences
        """
        kmer_counts = Counter()
        
        for seq in sequences:
            kmers = self.create_kmers(seq, k)
            kmer_counts.update(kmers)
        
        # Filter by frequency
        vocab = {kmer: idx for idx, (kmer, count) in enumerate(kmer_counts.items()) 
                if count >= min_freq}
        
        # Add special tokens
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        vocab['<MASK>'] = len(vocab)
        
        return vocab
    
    def align_sequences_pairwise(self, seq1, seq2):
        """
        Pairwise sequence alignment using Biopython
        """
        aligner = PairwiseAligner()
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
        
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]
        
        return {
            'score': best_alignment.score,
            'aligned_seq1': str(best_alignment).split('\n')[0],
            'aligned_seq2': str(best_alignment).split('\n')[2],
            'alignment_string': str(best_alignment).split('\n')[1]
        }
    
    def run_mafft_alignment(self, sequences, output_file=None):
        """
        Multiple sequence alignment using MAFFT
        Note: Requires MAFFT to be installed
        """
        if output_file is None:
            output_file = self.output_dir / "aligned_sequences.fasta"
        
        # Create temporary input file
        temp_input = self.output_dir / "temp_input.fasta"
        
        # Write sequences to temporary file
        with open(temp_input, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")
        
        try:
            # Run MAFFT (simplified - would need proper MAFFT installation)
            logger.info("MAFFT not available - using simple alignment placeholder")
            # In real implementation: subprocess.run(['mafft', str(temp_input)], ...)
            
            # For now, return input sequences
            aligned_sequences = sequences
            
        except Exception as e:
            logger.warning(f"MAFFT alignment failed: {e}, using input sequences")
            aligned_sequences = sequences
        
        # Clean up
        if temp_input.exists():
            temp_input.unlink()
        
        return aligned_sequences
    
    def create_mutation_mask(self, sequence, mutation_rate=0.1):
        """
        Create random mutation mask for training
        Similar to BERT's masked language modeling
        """
        seq_array = np.array(list(sequence))
        mask = np.random.random(len(seq_array)) < mutation_rate
        
        # Store original bases
        original_bases = seq_array[mask].copy()
        
        # Replace with random bases or N
        for i, should_mutate in enumerate(mask):
            if should_mutate:
                if np.random.random() < 0.8:  # 80% random base
                    seq_array[i] = np.random.choice(['A', 'T', 'G', 'C'])
                else:  # 20% unknown
                    seq_array[i] = 'N'
        
        return {
            'mutated_sequence': ''.join(seq_array),
            'mask': mask,
            'original_bases': original_bases,
            'mutation_positions': np.where(mask)[0]
        }
    
    def create_training_pairs(self, ancient_seq, modern_seq, window_size=1000):
        """
        Create training pairs from aligned ancient and modern sequences
        """
        training_pairs = []
        
        # Segment sequences
        for i in range(0, min(len(ancient_seq), len(modern_seq)) - window_size, window_size // 2):
            ancient_window = ancient_seq[i:i+window_size]
            modern_window = modern_seq[i:i+window_size]
            
            # Skip windows with too many N's
            if ancient_window.count('N') / len(ancient_window) > 0.5:
                continue
                
            training_pairs.append({
                'input': ancient_window,
                'target': modern_window,
                'position': i
            })
        
        return training_pairs
    
    def process_dataset(self, fasta_files, align=True):
        """
        Process entire dataset of FASTA files
        """
        logger.info("Processing ancient genome dataset...")
        
        all_sequences = []
        sequence_metadata = []
        
        # Load all sequences
        for fasta_file in fasta_files:
            logger.info(f"Processing {fasta_file}")
            
            for record in SeqIO.parse(fasta_file, "fasta"):
                cleaned_seq = self.clean_sequence(record.seq)
                
                if cleaned_seq:
                    # Segment long sequences
                    segments = self.segment_sequence(cleaned_seq)
                    
                    for segment in segments:
                        all_sequences.append(segment['sequence'])
                        sequence_metadata.append({
                            'original_id': record.id,
                            'species': record.id.split('_')[0],
                            'segment_start': segment['start'],
                            'segment_end': segment['end'],
                            'segment_length': segment['length'],
                            'source_file': fasta_file
                        })
        
        logger.info(f"Loaded {len(all_sequences)} sequence segments")
        
        # Create k-mer vocabulary
        logger.info("Creating k-mer vocabulary...")
        kmer_vocab = self.create_kmer_vocabulary(all_sequences)
        logger.info(f"K-mer vocabulary size: {len(kmer_vocab)}")
        
        # Align sequences if requested
        if align and len(all_sequences) > 1:
            logger.info("Aligning sequences...")
            aligned_sequences = self.run_mafft_alignment(all_sequences[:10])  # Sample for demo
        else:
            aligned_sequences = all_sequences
        
        # Create numerical representations
        logger.info("Creating numerical representations...")
        processed_data = {
            'sequences': all_sequences,
            'metadata': pd.DataFrame(sequence_metadata),
            'kmer_vocab': kmer_vocab,
            'aligned_sequences': aligned_sequences if align else None
        }
        
        # Create different encodings
        for encoding in ['integer', 'onehot']:
            logger.info(f"Creating {encoding} encoding...")
            encoded_sequences = []
            
            for seq in all_sequences[:100]:  # Sample for memory efficiency
                encoded = self.sequence_to_numerical(seq, encoding)
                encoded_sequences.append(encoded)
            
            processed_data[f'{encoding}_encoded'] = encoded_sequences
        
        # Save processed data
        output_file = self.output_dir / "processed_genome_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Processed data saved to {output_file}")
        
        return processed_data
    
    def create_masked_training_data(self, sequences, mask_ratio=0.15):
        """
        Create masked training data for BERT-style pre-training
        """
        training_examples = []
        
        for seq in sequences:
            # Create multiple masked versions of each sequence
            for _ in range(3):  # 3 different masks per sequence
                masked_data = self.create_mutation_mask(seq, mask_ratio)
                
                training_examples.append({
                    'input_sequence': masked_data['mutated_sequence'],
                    'target_sequence': seq,
                    'mask': masked_data['mask'],
                    'mutation_positions': masked_data['mutation_positions']
                })
        
        return training_examples
    
    def generate_statistics_report(self, processed_data):
        """
        Generate comprehensive statistics report
        """
        sequences = processed_data['sequences']
        metadata = processed_data['metadata']
        
        stats = {
            'total_sequences': len(sequences),
            'total_base_pairs': sum(len(seq) for seq in sequences),
            'average_length': np.mean([len(seq) for seq in sequences]),
            'length_std': np.std([len(seq) for seq in sequences]),
            'species_distribution': metadata['species'].value_counts().to_dict(),
            'gc_content_distribution': [],
            'n_content_distribution': []
        }
        
        # Calculate GC and N content for each sequence
        for seq in sequences[:1000]:  # Sample for efficiency
            gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
            n_content = seq.count('N') / len(seq) * 100
            stats['gc_content_distribution'].append(gc_content)
            stats['n_content_distribution'].append(n_content)
        
        stats['average_gc_content'] = np.mean(stats['gc_content_distribution'])
        stats['average_n_content'] = np.mean(stats['n_content_distribution'])
        
        # Save statistics
        stats_file = self.output_dir / "preprocessing_statistics.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Preprocessing statistics saved")
        return stats

def main():
    """
    Main preprocessing pipeline
    """
    preprocessor = DNAPreprocessor()
    
    # Example usage - in practice, use actual downloaded files
    from pathlib import Path
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        logger.error("Raw data directory not found. Run ancient_genome_downloader.py first.")
        return
    
    # Find FASTA files
    fasta_files = list(data_dir.glob("*.fasta")) + list(data_dir.glob("*.fa"))
    
    if not fasta_files:
        logger.error("No FASTA files found in data directory")
        return
    
    # Process dataset
    processed_data = preprocessor.process_dataset(fasta_files)
    
    # Generate statistics
    stats = preprocessor.generate_statistics_report(processed_data)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Processed {stats['total_sequences']} sequences")
    logger.info(f"Total base pairs: {stats['total_base_pairs']:,}")
    logger.info(f"Average GC content: {stats['average_gc_content']:.1f}%")
    logger.info(f"Average N content: {stats['average_n_content']:.1f}%")

if __name__ == "__main__":
    main()
