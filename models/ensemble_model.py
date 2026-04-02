"""
Ensemble Model for Genome Sequencing and Reconstruction

This module combines LSTM, Transformer, Autoencoder, and GNN models
for comprehensive DNA sequence reconstruction with confidence scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import pickle
import os

# Import our custom models
from lstm_model import DNASequencePredictor
from transformer_model import DNATransformer
from autoencoder_model import ConvolutionalDNAAutoencoder, TransformerDNAAutoencoder
from gnn_model import PhylogeneticGNN, PhylogeneticGraphBuilder, EnsemblePhylogeneticReconstructor

class DNAEnsemble(nn.Module):
    """
    Comprehensive ensemble model combining multiple approaches for DNA reconstruction
    """
    
    def __init__(self,
                 vocab_size: int = 5,
                 sequence_length: int = 1000,
                 ensemble_config: Optional[Dict] = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default ensemble configuration
        default_config = {
            'use_lstm': True,
            'use_transformer': True,
            'use_conv_autoencoder': True,
            'use_transformer_autoencoder': True,
            'use_gnn': True,
            'weights': {
                'lstm': 0.25,
                'transformer': 0.25,
                'conv_autoencoder': 0.15,
                'transformer_autoencoder': 0.15,
                'gnn': 0.20
            },
            'confidence_threshold': 0.7,
            'temperature': 0.8
        }
        
        self.config = ensemble_config if ensemble_config else default_config
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Ensemble combination layer
        self.ensemble_combiner = nn.Sequential(
            nn.Linear(vocab_size * len(self.models), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(vocab_size * len(self.models), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging for the ensemble"""
        logger = logging.getLogger('DNAEnsemble')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_models(self):
        """Initialize all component models"""
        
        if self.config.get('use_lstm', True):
            self.models['lstm'] = DNASequencePredictor(
                vocab_size=self.vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                max_length=self.sequence_length
            ).to(self.device)
            
        if self.config.get('use_transformer', True):
            self.models['transformer'] = DNATransformer(
                vocab_size=self.vocab_size,
                d_model=512,
                nhead=8,
                num_layers=6,
                max_length=self.sequence_length
            ).to(self.device)
            
        if self.config.get('use_conv_autoencoder', True):
            self.models['conv_autoencoder'] = ConvolutionalDNAAutoencoder(
                input_channels=self.vocab_size,
                sequence_length=self.sequence_length
            ).to(self.device)
            
        if self.config.get('use_transformer_autoencoder', True):
            self.models['transformer_autoencoder'] = TransformerDNAAutoencoder(
                vocab_size=self.vocab_size,
                d_model=512,
                nhead=8,
                num_layers=4,
                max_length=self.sequence_length
            ).to(self.device)
            
        if self.config.get('use_gnn', True):
            self.models['gnn'] = PhylogeneticGNN(
                node_feature_dim=100,
                hidden_dim=256,
                output_dim=512,
                num_layers=3,
                gnn_type='GCN'
            ).to(self.device)
            
        self.logger.info(f"Initialized ensemble with models: {list(self.models.keys())}")
        
    def forward(self, 
                input_sequence: torch.Tensor,
                phylogenetic_data: Optional = None,
                target_species_idx: int = 0,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            input_sequence: Input DNA sequence tensor [batch_size, seq_len]
            phylogenetic_data: Graph data for GNN (optional)
            target_species_idx: Target species index for GNN
            mask: Mask for missing positions
            
        Returns:
            Dictionary containing ensemble predictions and individual model outputs
        """
        batch_size = input_sequence.size(0)
        outputs = {}
        predictions = []
        confidences = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm':
                    # LSTM prediction
                    lstm_output = model(input_sequence)
                    lstm_probs = F.softmax(lstm_output, dim=-1)
                    outputs[model_name] = lstm_probs
                    predictions.append(lstm_probs.mean(dim=1))  # Average over sequence length
                    
                elif model_name == 'transformer':
                    # Transformer prediction
                    if mask is not None:
                        transformer_output = model(input_sequence, mask=mask)
                    else:
                        transformer_output = model(input_sequence)
                    transformer_probs = F.softmax(transformer_output, dim=-1)
                    outputs[model_name] = transformer_probs
                    predictions.append(transformer_probs.mean(dim=1))
                    
                elif 'autoencoder' in model_name:
                    # Autoencoder denoising
                    one_hot_input = F.one_hot(input_sequence, self.vocab_size).float()
                    if model_name == 'conv_autoencoder':
                        one_hot_input = one_hot_input.permute(0, 2, 1)  # [batch, channels, seq_len]
                    
                    denoised = model(one_hot_input)
                    
                    if model_name == 'conv_autoencoder':
                        denoised = denoised.permute(0, 2, 1)  # Back to [batch, seq_len, vocab]
                        
                    autoencoder_probs = F.softmax(denoised, dim=-1)
                    outputs[model_name] = autoencoder_probs
                    predictions.append(autoencoder_probs.mean(dim=1))
                    
                elif model_name == 'gnn' and phylogenetic_data is not None:
                    # GNN prediction
                    gnn_logits = model(phylogenetic_data, target_species_idx)
                    gnn_probs = F.softmax(gnn_logits, dim=-1).unsqueeze(0).expand(batch_size, -1)
                    outputs[model_name] = gnn_probs
                    predictions.append(gnn_probs)
                    
            except Exception as e:
                self.logger.warning(f"Error in {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise RuntimeError("No models produced valid predictions")
            
        # Combine predictions
        stacked_predictions = torch.stack(predictions, dim=-1)  # [batch, vocab, num_models]
        concatenated = stacked_predictions.view(batch_size, -1)  # [batch, vocab * num_models]
        
        # Ensemble combination
        ensemble_logits = self.ensemble_combiner(concatenated)
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        
        # Confidence prediction
        confidence_scores = self.confidence_predictor(concatenated)
        
        return {
            'ensemble_prediction': ensemble_probs,
            'confidence_scores': confidence_scores,
            'individual_outputs': outputs,
            'raw_predictions': stacked_predictions
        }
    
    def reconstruct_sequence(self,
                           incomplete_sequence: Union[str, torch.Tensor],
                           target_length: Optional[int] = None,
                           phylogenetic_data: Optional = None,
                           target_species_idx: int = 0,
                           use_iterative: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Reconstruct incomplete DNA sequence using ensemble
        
        Args:
            incomplete_sequence: Input sequence (string or tensor)
            target_length: Desired output length
            phylogenetic_data: Phylogenetic graph data
            target_species_idx: Target species index
            use_iterative: Whether to use iterative reconstruction
            
        Returns:
            Tuple of (reconstructed_sequence, confidence_metrics)
        """
        self.eval()
        
        # Convert to tensor if needed
        if isinstance(incomplete_sequence, str):
            sequence_tensor = self._sequence_to_tensor(incomplete_sequence)
        else:
            sequence_tensor = incomplete_sequence
            
        if target_length is None:
            target_length = self.sequence_length
            
        # Pad or truncate to target length
        current_length = sequence_tensor.size(0)
        if current_length < target_length:
            # Pad with N tokens (index 4)
            padding = torch.full((target_length - current_length,), 4, dtype=sequence_tensor.dtype)
            sequence_tensor = torch.cat([sequence_tensor, padding])
        else:
            sequence_tensor = sequence_tensor[:target_length]
            
        sequence_tensor = sequence_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Create mask for missing positions
        mask = (sequence_tensor == 4).float()  # N tokens are missing
        
        if use_iterative:
            reconstructed = self._iterative_reconstruction(
                sequence_tensor, mask, phylogenetic_data, target_species_idx
            )
        else:
            with torch.no_grad():
                outputs = self.forward(
                    sequence_tensor, phylogenetic_data, target_species_idx, mask
                )
                reconstructed = sequence_tensor.clone()
                
                # Fill missing positions with ensemble predictions
                missing_positions = (mask[0] == 1)
                if missing_positions.any():
                    predicted_bases = torch.argmax(outputs['ensemble_prediction'], dim=-1)
                    reconstructed[0, missing_positions] = predicted_bases[0].expand_as(reconstructed[0, missing_positions])
        
        # Convert back to string
        reconstructed_sequence = self._tensor_to_sequence(reconstructed[0])
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            reconstructed, mask, phylogenetic_data, target_species_idx
        )
        
        return reconstructed_sequence, confidence_metrics
    
    def _iterative_reconstruction(self,
                                sequence_tensor: torch.Tensor,
                                mask: torch.Tensor,
                                phylogenetic_data: Optional,
                                target_species_idx: int,
                                max_iterations: int = 10) -> torch.Tensor:
        """Iteratively reconstruct sequence with confidence-based updates"""
        
        reconstructed = sequence_tensor.clone()
        current_mask = mask.clone()
        
        for iteration in range(max_iterations):
            with torch.no_grad():
                outputs = self.forward(
                    reconstructed, phylogenetic_data, target_species_idx, current_mask
                )
                
                ensemble_pred = outputs['ensemble_prediction']
                confidence = outputs['confidence_scores']
                
                # Update positions with high confidence
                missing_positions = (current_mask[0] == 1)
                if not missing_positions.any():
                    break
                    
                high_confidence = confidence[0] > self.config['confidence_threshold']
                update_positions = missing_positions & high_confidence
                
                if update_positions.any():
                    predicted_bases = torch.argmax(ensemble_pred, dim=-1)
                    reconstructed[0, update_positions] = predicted_bases[0].expand_as(reconstructed[0, update_positions])
                    current_mask[0, update_positions] = 0
                else:
                    # Lower threshold if no high-confidence predictions
                    lower_threshold = max(0.5, self.config['confidence_threshold'] - 0.1)
                    update_positions = missing_positions & (confidence[0] > lower_threshold)
                    if update_positions.any():
                        predicted_bases = torch.argmax(ensemble_pred, dim=-1)
                        reconstructed[0, update_positions] = predicted_bases[0].expand_as(reconstructed[0, update_positions])
                        current_mask[0, update_positions] = 0
                    else:
                        break
                        
            self.logger.info(f"Iteration {iteration + 1}: {update_positions.sum().item()} positions updated")
            
        return reconstructed
    
    def _calculate_confidence_metrics(self,
                                    sequence_tensor: torch.Tensor,
                                    original_mask: torch.Tensor,
                                    phylogenetic_data: Optional,
                                    target_species_idx: int) -> Dict[str, float]:
        """Calculate various confidence metrics for the reconstruction"""
        
        with torch.no_grad():
            outputs = self.forward(
                sequence_tensor, phylogenetic_data, target_species_idx, torch.zeros_like(original_mask)
            )
            
            ensemble_pred = outputs['ensemble_prediction']
            confidence_scores = outputs['confidence_scores']
            
            metrics = {
                'overall_confidence': float(confidence_scores.mean()),
                'max_probability': float(torch.max(ensemble_pred)),
                'entropy': float(-torch.sum(ensemble_pred * torch.log(ensemble_pred + 1e-8), dim=-1).mean()),
                'prediction_consensus': self._calculate_consensus(outputs['individual_outputs'])
            }
            
            # Calculate per-base confidence for missing positions
            missing_positions = (original_mask[0] == 1)
            if missing_positions.any():
                reconstructed_confidence = confidence_scores[0, missing_positions] if confidence_scores.dim() > 1 else confidence_scores[0]
                metrics['reconstructed_positions_confidence'] = float(reconstructed_confidence.mean())
            else:
                metrics['reconstructed_positions_confidence'] = 1.0
                
        return metrics
    
    def _calculate_consensus(self, individual_outputs: Dict[str, torch.Tensor]) -> float:
        """Calculate consensus between individual model predictions"""
        if len(individual_outputs) < 2:
            return 1.0
            
        predictions = []
        for model_name, output in individual_outputs.items():
            if output.dim() > 2:
                pred = torch.argmax(output.mean(dim=1), dim=-1)
            else:
                pred = torch.argmax(output, dim=-1)
            predictions.append(pred)
            
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreement = (predictions[i] == predictions[j]).float().mean()
                agreements.append(agreement)
                
        return float(torch.mean(torch.stack(agreements)))
    
    def _sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence string to tensor"""
        base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        return torch.tensor([base_map.get(base.upper(), 4) for base in sequence], dtype=torch.long)
    
    def _tensor_to_sequence(self, tensor: torch.Tensor) -> str:
        """Convert tensor to DNA sequence string"""
        base_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        return ''.join([base_map[int(idx)] for idx in tensor.cpu().numpy()])
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble model"""
        checkpoint = {
            'ensemble_state': self.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'model_names': list(self.models.keys())
        }
        
        # Save individual model states
        for name, model in self.models.items():
            checkpoint[f'{name}_state'] = model.state_dict()
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load the entire ensemble model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']
        self.sequence_length = checkpoint['sequence_length']
        
        # Reinitialize models
        self._initialize_models()
        
        # Load individual model states
        for name in checkpoint['model_names']:
            if name in self.models and f'{name}_state' in checkpoint:
                self.models[name].load_state_dict(checkpoint[f'{name}_state'])
                
        # Load ensemble state
        self.load_state_dict(checkpoint['ensemble_state'])
        
        self.logger.info(f"Ensemble loaded from {filepath}")


class DNAReconstructionPipeline:
    """Complete pipeline for DNA sequence reconstruction"""
    
    def __init__(self, ensemble_config: Optional[Dict] = None):
        self.ensemble = DNAEnsemble(ensemble_config=ensemble_config)
        self.phylo_builder = PhylogeneticGraphBuilder()
        self.logger = logging.getLogger('DNAReconstructionPipeline')
    
    def process_ancient_genome(self,
                             species_sequences: Dict[str, str],
                             target_species: str,
                             incomplete_sequence: str,
                             target_length: Optional[int] = None) -> Dict[str, any]:
        """
        Complete pipeline for processing ancient genome data
        
        Args:
            species_sequences: Dictionary of related species sequences
            target_species: Name of target species to reconstruct
            incomplete_sequence: Partial sequence to complete
            target_length: Desired output length
            
        Returns:
            Dictionary containing reconstruction results and analysis
        """
        
        self.logger.info(f"Starting reconstruction for {target_species}")
        
        # Build phylogenetic graph
        phylo_data = self.phylo_builder.create_phylogenetic_graph(species_sequences)
        
        # Find target species index
        target_idx = None
        for idx, species in phylo_data.species_mapping.items():
            if species == target_species:
                target_idx = idx
                break
                
        if target_idx is None:
            raise ValueError(f"Target species {target_species} not found")
        
        # Reconstruct sequence
        reconstructed_seq, confidence_metrics = self.ensemble.reconstruct_sequence(
            incomplete_sequence=incomplete_sequence,
            target_length=target_length,
            phylogenetic_data=phylo_data,
            target_species_idx=target_idx,
            use_iterative=True
        )
        
        # Analyze reconstruction
        analysis = self._analyze_reconstruction(
            original=incomplete_sequence,
            reconstructed=reconstructed_seq,
            confidence=confidence_metrics
        )
        
        results = {
            'target_species': target_species,
            'original_sequence': incomplete_sequence,
            'reconstructed_sequence': reconstructed_seq,
            'confidence_metrics': confidence_metrics,
            'analysis': analysis,
            'phylogenetic_info': {
                'num_related_species': len(species_sequences),
                'graph_nodes': phylo_data.x.shape[0],
                'graph_edges': phylo_data.edge_index.shape[1]
            }
        }
        
        self.logger.info("Reconstruction completed successfully")
        return results
    
    def _analyze_reconstruction(self, 
                              original: str, 
                              reconstructed: str, 
                              confidence: Dict[str, float]) -> Dict[str, any]:
        """Analyze reconstruction quality and statistics"""
        
        analysis = {
            'sequence_statistics': {
                'original_length': len(original),
                'reconstructed_length': len(reconstructed),
                'missing_positions': original.count('N'),
                'reconstruction_rate': 1.0 - (original.count('N') / len(original))
            },
            'base_composition': self._calculate_base_composition(reconstructed),
            'quality_metrics': confidence,
            'mutation_analysis': self._analyze_mutations(original, reconstructed)
        }
        
        return analysis
    
    def _calculate_base_composition(self, sequence: str) -> Dict[str, float]:
        """Calculate base composition percentages"""
        total = len(sequence)
        if total == 0:
            return {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
            
        return {
            'A': sequence.count('A') / total,
            'T': sequence.count('T') / total,
            'G': sequence.count('G') / total,
            'C': sequence.count('C') / total,
            'N': sequence.count('N') / total
        }
    
    def _analyze_mutations(self, original: str, reconstructed: str) -> Dict[str, any]:
        """Analyze potential mutations in the reconstruction"""
        
        mutations = []
        min_len = min(len(original), len(reconstructed))
        
        for i in range(min_len):
            if original[i] != 'N' and original[i] != reconstructed[i]:
                mutations.append({
                    'position': i,
                    'original': original[i],
                    'reconstructed': reconstructed[i],
                    'type': self._classify_mutation(original[i], reconstructed[i])
                })
        
        return {
            'total_mutations': len(mutations),
            'mutation_rate': len(mutations) / min_len if min_len > 0 else 0,
            'mutations': mutations[:50],  # Limit to first 50 for display
            'mutation_types': self._count_mutation_types(mutations)
        }
    
    def _classify_mutation(self, original: str, mutated: str) -> str:
        """Classify type of mutation"""
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        if (original, mutated) in transitions:
            return 'transition'
        else:
            return 'transversion'
    
    def _count_mutation_types(self, mutations: List[Dict]) -> Dict[str, int]:
        """Count different types of mutations"""
        counts = {'transition': 0, 'transversion': 0}
        for mut in mutations:
            counts[mut['type']] += 1
        return counts


# Example usage and testing
if __name__ == "__main__":
    
    print("Testing DNA Ensemble Model...")
    
    # Example species data
    species_sequences = {
        'Human': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC' * 10,
        'Neanderthal': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATT' * 10,
        'Chimpanzee': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAAC' * 10,
        'Mammoth': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAGG' * 10
    }
    
    # Incomplete sequence (with N's representing missing bases)
    incomplete_seq = 'ATGCGATCGATCGATCGATCGATCGATCGATNNNNNNNNNGATCGATCGATCGATCGATCGATC'
    
    # Initialize pipeline
    pipeline = DNAReconstructionPipeline()
    
    # Process the reconstruction
    results = pipeline.process_ancient_genome(
        species_sequences=species_sequences,
        target_species='Neanderthal',
        incomplete_sequence=incomplete_seq,
        target_length=200
    )
    
    # Display results
    print(f"\n=== Reconstruction Results for {results['target_species']} ===")
    print(f"Original:      {results['original_sequence']}")
    print(f"Reconstructed: {results['reconstructed_sequence'][:len(results['original_sequence'])]}")
    
    print(f"\n=== Confidence Metrics ===")
    for metric, value in results['confidence_metrics'].items():
        print(f"{metric}: {value:.3f}")
    
    print(f"\n=== Analysis ===")
    analysis = results['analysis']
    print(f"Missing positions: {analysis['sequence_statistics']['missing_positions']}")
    print(f"Reconstruction rate: {analysis['sequence_statistics']['reconstruction_rate']:.3f}")
    print(f"Total mutations: {analysis['mutation_analysis']['total_mutations']}")
    print(f"Mutation rate: {analysis['mutation_analysis']['mutation_rate']:.3f}")
    
    print("\nEnsemble model testing completed successfully!")
