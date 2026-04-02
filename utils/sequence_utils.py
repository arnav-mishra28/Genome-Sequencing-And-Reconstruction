"""
Utility Functions for Genome Sequencing and Reconstruction

This module provides helper functions for data processing, visualization,
evaluation metrics, and common operations across the framework.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import Counter
import re
import json
import pickle
from datetime import datetime
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class DNASequenceUtils:
    """Utilities for DNA sequence manipulation and analysis"""
    
    @staticmethod
    def validate_dna_sequence(sequence: str) -> bool:
        """Validate if string contains only valid DNA bases"""
        valid_bases = set('ATGCN')
        return all(base.upper() in valid_bases for base in sequence)
    
    @staticmethod
    def clean_sequence(sequence: str, 
                      remove_ambiguous: bool = False,
                      standardize_case: bool = True) -> str:
        """Clean and standardize DNA sequence"""
        if standardize_case:
            sequence = sequence.upper()
            
        # Remove non-DNA characters
        sequence = re.sub(r'[^ATGCN]', '', sequence)
        
        if remove_ambiguous:
            sequence = re.sub(r'N', '', sequence)
            
        return sequence
    
    @staticmethod
    def complement(sequence: str) -> str:
        """Get complement of DNA sequence"""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement_map.get(base, base) for base in sequence.upper())
    
    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get reverse complement of DNA sequence"""
        return DNASequenceUtils.complement(sequence)[::-1]
    
    @staticmethod
    def translate_to_protein(sequence: str, frame: int = 0) -> str:
        """Translate DNA sequence to protein sequence"""
        # Genetic code table
        genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        sequence = sequence[frame:].upper()
        protein = ''
        
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if len(codon) == 3:
                amino_acid = genetic_code.get(codon, 'X')
                protein += amino_acid
                
        return protein
    
    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content of DNA sequence"""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_count = len([base for base in sequence if base in 'ATGC'])
        
        return gc_count / total_count if total_count > 0 else 0.0
    
    @staticmethod
    def find_orfs(sequence: str, min_length: int = 100) -> List[Dict]:
        """Find open reading frames (ORFs) in sequence"""
        orfs = []
        
        for frame in range(3):
            for strand in [sequence, DNASequenceUtils.reverse_complement(sequence)]:
                start_positions = []
                
                # Find start codons (ATG)
                for i in range(frame, len(strand) - 2, 3):
                    codon = strand[i:i+3]
                    if codon == 'ATG':
                        start_positions.append(i)
                
                # Find stop codons for each start position
                for start in start_positions:
                    for i in range(start + 3, len(strand) - 2, 3):
                        codon = strand[i:i+3]
                        if codon in ['TAA', 'TAG', 'TGA']:
                            orf_length = i - start + 3
                            if orf_length >= min_length:
                                orfs.append({
                                    'start': start,
                                    'end': i + 3,
                                    'length': orf_length,
                                    'frame': frame,
                                    'strand': 'forward' if strand == sequence else 'reverse',
                                    'sequence': strand[start:i+3],
                                    'protein': DNASequenceUtils.translate_to_protein(strand[start:i+3])
                                })
                            break
                            
        return sorted(orfs, key=lambda x: x['length'], reverse=True)


class SequenceEncoder:
    """Encode DNA sequences for machine learning models"""
    
    def __init__(self, encoding_type: str = 'integer'):
        self.encoding_type = encoding_type
        self.base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.reverse_map = {v: k for k, v in self.base_map.items()}
        
    def encode_sequence(self, sequence: str, max_length: Optional[int] = None) -> np.ndarray:
        """Encode DNA sequence to numerical format"""
        sequence = sequence.upper()
        
        if self.encoding_type == 'integer':
            encoded = np.array([self.base_map.get(base, 4) for base in sequence])
            
        elif self.encoding_type == 'one_hot':
            encoded = np.zeros((len(sequence), len(self.base_map)))
            for i, base in enumerate(sequence):
                base_idx = self.base_map.get(base, 4)
                encoded[i, base_idx] = 1
                
        elif self.encoding_type == 'kmer':
            k = 6  # hexamer encoding
            encoded = self._kmer_encode(sequence, k)
            
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Pad or truncate if max_length specified
        if max_length is not None:
            if self.encoding_type == 'integer':
                if len(encoded) < max_length:
                    encoded = np.pad(encoded, (0, max_length - len(encoded)), constant_values=4)
                else:
                    encoded = encoded[:max_length]
            elif self.encoding_type == 'one_hot':
                if encoded.shape[0] < max_length:
                    padding = np.zeros((max_length - encoded.shape[0], encoded.shape[1]))
                    padding[:, 4] = 1  # N padding
                    encoded = np.vstack([encoded, padding])
                else:
                    encoded = encoded[:max_length]
                    
        return encoded
    
    def decode_sequence(self, encoded: np.ndarray) -> str:
        """Decode numerical format back to DNA sequence"""
        if self.encoding_type == 'integer':
            return ''.join([self.reverse_map.get(int(idx), 'N') for idx in encoded])
        elif self.encoding_type == 'one_hot':
            indices = np.argmax(encoded, axis=1)
            return ''.join([self.reverse_map.get(int(idx), 'N') for idx in indices])
        else:
            raise NotImplementedError(f"Decoding not implemented for {self.encoding_type}")
    
    def _kmer_encode(self, sequence: str, k: int) -> np.ndarray:
        """Encode sequence using k-mer representation"""
        kmers = {}
        kmer_idx = 0
        
        # Generate all possible k-mers
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if kmer not in kmers:
                kmers[kmer] = kmer_idx
                kmer_idx += 1
        
        # Encode sequence as k-mer indices
        encoded = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            encoded.append(kmers[kmer])
            
        return np.array(encoded)


class ModelEvaluator:
    """Evaluate model performance and generate metrics"""
    
    @staticmethod
    def calculate_base_accuracy(predictions: np.ndarray, targets: np.ndarray, 
                              ignore_n: bool = True) -> float:
        """Calculate per-base accuracy"""
        if ignore_n:
            # Ignore N bases (index 4) in evaluation
            mask = targets != 4
            if np.sum(mask) == 0:
                return 0.0
            return accuracy_score(targets[mask], predictions[mask])
        else:
            return accuracy_score(targets, predictions)
    
    @staticmethod
    def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
        """Calculate sequence similarity percentage"""
        if len(seq1) != len(seq2):
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
            
        if len(seq1) == 0:
            return 0.0
            
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != 'N')
        total_valid = sum(1 for a, b in zip(seq1, seq2) if a != 'N' and b != 'N')
        
        return matches / total_valid if total_valid > 0 else 0.0
    
    @staticmethod
    def evaluate_reconstruction_quality(original: str, reconstructed: str, 
                                      missing_positions: List[int]) -> Dict[str, float]:
        """Evaluate quality of sequence reconstruction"""
        metrics = {}
        
        # Overall similarity
        metrics['overall_similarity'] = ModelEvaluator.calculate_sequence_similarity(
            original, reconstructed
        )
        
        # Accuracy only on reconstructed positions
        if missing_positions:
            original_missing = ''.join([original[i] for i in missing_positions if i < len(original)])
            reconstructed_missing = ''.join([reconstructed[i] for i in missing_positions if i < len(reconstructed)])
            metrics['reconstruction_accuracy'] = ModelEvaluator.calculate_sequence_similarity(
                original_missing, reconstructed_missing
            )
        else:
            metrics['reconstruction_accuracy'] = 1.0
        
        # GC content preservation
        original_gc = DNASequenceUtils.calculate_gc_content(original)
        reconstructed_gc = DNASequenceUtils.calculate_gc_content(reconstructed)
        metrics['gc_content_preservation'] = 1.0 - abs(original_gc - reconstructed_gc)
        
        return metrics


class VisualizationUtils:
    """Utilities for visualizing results and analysis"""
    
    @staticmethod
    def plot_base_composition(sequences: Dict[str, str], title: str = "Base Composition"):
        """Plot base composition comparison"""
        compositions = {}
        bases = ['A', 'T', 'G', 'C', 'N']
        
        for name, seq in sequences.items():
            comp = {}
            total = len(seq)
            for base in bases:
                comp[base] = seq.count(base) / total if total > 0 else 0
            compositions[name] = comp
        
        df = pd.DataFrame(compositions).T
        
        plt.figure(figsize=(10, 6))
        df[['A', 'T', 'G', 'C']].plot(kind='bar', stacked=False)
        plt.title(title)
        plt.xlabel('Sequences')
        plt.ylabel('Proportion')
        plt.legend(title='Bases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sequence_similarity_matrix(sequences: Dict[str, str]):
        """Plot similarity matrix between sequences"""
        names = list(sequences.keys())
        n = len(names)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = ModelEvaluator.calculate_sequence_similarity(
                        sequences[names[i]], sequences[names[j]]
                    )
                    similarity_matrix[i, j] = sim
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, 
                   xticklabels=names, 
                   yticklabels=names,
                   annot=True, 
                   cmap='viridis',
                   vmin=0, vmax=1)
        plt.title('Sequence Similarity Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_reconstruction_confidence(confidence_scores: List[float], 
                                     positions: List[int] = None):
        """Plot confidence scores along sequence positions"""
        if positions is None:
            positions = list(range(len(confidence_scores)))
        
        plt.figure(figsize=(12, 4))
        plt.plot(positions, confidence_scores, alpha=0.7)
        plt.fill_between(positions, confidence_scores, alpha=0.3)
        plt.xlabel('Position')
        plt.ylabel('Confidence Score')
        plt.title('Reconstruction Confidence Along Sequence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gc_content_distribution(sequences: Dict[str, str], window_size: int = 100):
        """Plot GC content distribution along sequences"""
        plt.figure(figsize=(12, 6))
        
        for name, seq in sequences.items():
            gc_contents = []
            positions = []
            
            for i in range(0, len(seq) - window_size + 1, window_size // 2):
                window = seq[i:i + window_size]
                gc = DNASequenceUtils.calculate_gc_content(window)
                gc_contents.append(gc)
                positions.append(i + window_size // 2)
            
            plt.plot(positions, gc_contents, label=name, alpha=0.7)
        
        plt.xlabel('Position')
        plt.ylabel('GC Content')
        plt.title(f'GC Content Distribution (Window size: {window_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class DataLogger:
    """Log and save experimental results"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, experiment_name: str, config: Dict = None):
        """Start logging a new experiment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment = f"{experiment_name}_{timestamp}"
        
        experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration
        if config:
            config_path = os.path.join(experiment_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Started experiment: {self.current_experiment}")
        
    def log_results(self, results: Dict, filename: str = 'results.json'):
        """Log experiment results"""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        results_path = os.path.join(experiment_dir, filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {results_path}")
        
    def save_sequence(self, sequence: str, filename: str, description: str = ""):
        """Save DNA sequence to file"""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
            
        experiment_dir = os.path.join(self.log_dir, self.current_experiment)
        sequence_path = os.path.join(experiment_dir, filename)
        
        with open(sequence_path, 'w') as f:
            if description:
                f.write(f"# {description}\n")
            f.write(f">{filename}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')
        
        print(f"Sequence saved to {sequence_path}")


class MutationAnalyzer:
    """Analyze mutations and their effects"""
    
    @staticmethod
    def identify_mutations(reference: str, query: str) -> List[Dict]:
        """Identify mutations between two sequences"""
        mutations = []
        min_len = min(len(reference), len(query))
        
        for i in range(min_len):
            if reference[i] != query[i] and reference[i] != 'N' and query[i] != 'N':
                mutations.append({
                    'position': i,
                    'reference': reference[i],
                    'query': query[i],
                    'type': MutationAnalyzer.classify_mutation(reference[i], query[i])
                })
        
        return mutations
    
    @staticmethod
    def classify_mutation(ref_base: str, query_base: str) -> str:
        """Classify mutation type"""
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        if (ref_base, query_base) in transitions:
            return 'transition'
        else:
            return 'transversion'
    
    @staticmethod
    def calculate_mutation_rate(reference: str, query: str) -> float:
        """Calculate mutation rate between sequences"""
        mutations = MutationAnalyzer.identify_mutations(reference, query)
        valid_positions = sum(1 for i in range(min(len(reference), len(query))) 
                            if reference[i] != 'N' and query[i] != 'N')
        
        return len(mutations) / valid_positions if valid_positions > 0 else 0.0
    
    @staticmethod
    def analyze_codon_impact(sequence: str, mutations: List[Dict]) -> List[Dict]:
        """Analyze impact of mutations on codons and amino acids"""
        codon_impacts = []
        
        for mutation in mutations:
            pos = mutation['position']
            codon_start = (pos // 3) * 3
            codon_pos = pos % 3
            
            if codon_start + 2 < len(sequence):
                # Original codon
                original_codon = sequence[codon_start:codon_start + 3]
                
                # Mutated codon
                mutated_codon = list(original_codon)
                mutated_codon[codon_pos] = mutation['query']
                mutated_codon = ''.join(mutated_codon)
                
                # Translate codons
                original_aa = DNASequenceUtils.translate_to_protein(original_codon)
                mutated_aa = DNASequenceUtils.translate_to_protein(mutated_codon)
                
                impact = {
                    'position': pos,
                    'codon_position': codon_pos,
                    'original_codon': original_codon,
                    'mutated_codon': mutated_codon,
                    'original_aa': original_aa,
                    'mutated_aa': mutated_aa,
                    'effect': 'synonymous' if original_aa == mutated_aa else 'non-synonymous'
                }
                
                codon_impacts.append(impact)
        
        return codon_impacts


# Example usage and testing functions
def run_utility_examples():
    """Run examples demonstrating utility functions"""
    
    print("=== DNA Sequence Utilities Examples ===")
    
    # Example sequence
    test_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    
    # Basic utilities
    print(f"Original sequence: {test_sequence}")
    print(f"GC content: {DNASequenceUtils.calculate_gc_content(test_sequence):.2f}")
    print(f"Reverse complement: {DNASequenceUtils.reverse_complement(test_sequence)}")
    
    # Translation
    protein = DNASequenceUtils.translate_to_protein(test_sequence)
    print(f"Protein translation: {protein}")
    
    # Encoding
    encoder = SequenceEncoder(encoding_type='integer')
    encoded = encoder.encode_sequence(test_sequence, max_length=100)
    decoded = encoder.decode_sequence(encoded)
    print(f"Encoding test - matches original: {decoded[:len(test_sequence)] == test_sequence}")
    
    # Mutation analysis
    mutated_seq = test_sequence[:20] + "TTTT" + test_sequence[24:]
    mutations = MutationAnalyzer.identify_mutations(test_sequence, mutated_seq)
    mutation_rate = MutationAnalyzer.calculate_mutation_rate(test_sequence, mutated_seq)
    print(f"Mutations found: {len(mutations)}")
    print(f"Mutation rate: {mutation_rate:.4f}")
    
    # Evaluation
    similarity = ModelEvaluator.calculate_sequence_similarity(test_sequence, mutated_seq)
    print(f"Sequence similarity: {similarity:.3f}")
    
    print("\nUtility functions working correctly!")


if __name__ == "__main__":
    run_utility_examples()
