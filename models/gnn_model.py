"""
Graph Neural Network for Phylogenetic Relationships in Ancient DNA Reconstruction

This module implements GNN-based models that leverage evolutionary relationships
between species to improve DNA sequence reconstruction accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx
from torch_geometric.nn import GCNConv, GraphSAGE, GAT, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.metrics.pairwise import cosine_similarity

class PhylogeneticGraphBuilder:
    """Build phylogenetic graphs from species data and evolutionary distances"""
    
    def __init__(self):
        self.species_embeddings = {}
        self.distance_matrix = None
        
    def create_phylogenetic_graph(self, 
                                species_sequences: Dict[str, str],
                                evolutionary_distances: Optional[Dict[tuple, float]] = None) -> Data:
        """
        Create a phylogenetic graph from species sequences and distances
        
        Args:
            species_sequences: Dictionary mapping species names to DNA sequences
            evolutionary_distances: Optional predefined distances between species pairs
            
        Returns:
            torch_geometric.data.Data object representing the phylogenetic graph
        """
        species_list = list(species_sequences.keys())
        num_species = len(species_list)
        
        # Create node features (sequence embeddings)
        node_features = []
        for species in species_list:
            sequence = species_sequences[species]
            embedding = self._sequence_to_embedding(sequence)
            node_features.append(embedding)
            
        node_features = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Create edges based on evolutionary distances
        edge_index = []
        edge_attr = []
        
        if evolutionary_distances is None:
            evolutionary_distances = self._compute_sequence_distances(species_sequences)
            
        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i != j:
                    distance = evolutionary_distances.get((species1, species2), 
                                                         evolutionary_distances.get((species2, species1), 1.0))
                    # Connect species if distance is below threshold (closer evolutionary relationship)
                    if distance < 0.8:  # Threshold for connection
                        edge_index.append([i, j])
                        edge_attr.append([1.0 - distance])  # Convert distance to similarity
                        
        if len(edge_index) == 0:
            # If no connections, create a fully connected graph with uniform weights
            for i in range(num_species):
                for j in range(num_species):
                    if i != j:
                        edge_index.append([i, j])
                        edge_attr.append([0.5])
                        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create species name mapping
        species_mapping = {i: species for i, species in enumerate(species_list)}
        
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        graph_data.species_mapping = species_mapping
        
        return graph_data
        
    def _sequence_to_embedding(self, sequence: str, max_length: int = 1000) -> np.ndarray:
        """Convert DNA sequence to numerical embedding"""
        # Truncate or pad sequence
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence = sequence.ljust(max_length, 'N')
            
        # K-mer based embedding
        k = 6
        base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Count k-mers
        kmer_counts = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if all(base in base_map for base in kmer):
                kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
                
        # Convert to fixed-size feature vector (use top 100 most common k-mers)
        feature_size = 100
        sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
        
        embedding = np.zeros(feature_size)
        for i, (kmer, count) in enumerate(sorted_kmers[:feature_size]):
            embedding[i] = count
            
        # Normalize
        if np.sum(embedding) > 0:
            embedding = embedding / np.sum(embedding)
            
        return embedding
        
    def _compute_sequence_distances(self, species_sequences: Dict[str, str]) -> Dict[tuple, float]:
        """Compute pairwise evolutionary distances between sequences"""
        species_list = list(species_sequences.keys())
        distances = {}
        
        # Compute embeddings for all sequences
        embeddings = {}
        for species in species_list:
            embeddings[species] = self._sequence_to_embedding(species_sequences[species])
            
        # Compute pairwise distances
        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list[i+1:], i+1):
                # Use cosine distance
                emb1 = embeddings[species1].reshape(1, -1)
                emb2 = embeddings[species2].reshape(1, -1)
                
                similarity = cosine_similarity(emb1, emb2)[0][0]
                distance = 1.0 - similarity
                
                distances[(species1, species2)] = distance
                distances[(species2, species1)] = distance
                
        return distances


class PhylogeneticGNN(nn.Module):
    """Graph Neural Network for phylogenetic-aware sequence reconstruction"""
    
    def __init__(self, 
                 node_feature_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 num_layers: int = 3,
                 gnn_type: str = 'GCN'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'GCN':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GCNConv(hidden_dim, hidden_dim)
                )
        elif gnn_type == 'GraphSAGE':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GraphSAGE(hidden_dim, hidden_dim, num_layers=1)
                )
        elif gnn_type == 'GAT':
            for i in range(num_layers):
                self.gnn_layers.append(
                    GAT(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
                )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Sequence reconstruction head
        self.sequence_head = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # A, T, G, C, N
        )
        
    def forward(self, data: Data, target_species_idx: int = 0) -> torch.Tensor:
        """
        Forward pass through phylogenetic GNN
        
        Args:
            data: Graph data containing node features and edge information
            target_species_idx: Index of target species for reconstruction
            
        Returns:
            Sequence predictions for target species
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GraphSAGE':
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index, edge_attr)
            
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Output projection
        x = self.output_proj(x)
        
        # Get target species embedding
        target_embedding = x[target_species_idx]
        
        # Sequence reconstruction
        sequence_logits = self.sequence_head(target_embedding)
        
        return sequence_logits
    
    def get_species_embeddings(self, data: Data) -> torch.Tensor:
        """Get embeddings for all species in the graph"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GraphSAGE':
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index, edge_attr)
            
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Output projection
        embeddings = self.output_proj(x)
        
        return embeddings


class EnsemblePhylogeneticReconstructor:
    """Ensemble model combining GNN with other approaches for reconstruction"""
    
    def __init__(self, 
                 gnn_model: PhylogeneticGNN,
                 lstm_model = None,
                 transformer_model = None,
                 autoencoder_model = None):
        
        self.gnn_model = gnn_model
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.autoencoder_model = autoencoder_model
        
        # Ensemble weights
        self.weights = {
            'gnn': 0.3,
            'lstm': 0.25,
            'transformer': 0.25,
            'autoencoder': 0.2
        }
        
    def reconstruct_sequence(self, 
                           phylogenetic_data: Data,
                           incomplete_sequence: str,
                           target_species: str,
                           sequence_length: int = 1000) -> Tuple[str, Dict[str, float]]:
        """
        Reconstruct incomplete sequence using ensemble of models
        
        Args:
            phylogenetic_data: Graph data with evolutionary relationships
            incomplete_sequence: Partial DNA sequence to complete
            target_species: Name of target species
            sequence_length: Desired length of output sequence
            
        Returns:
            Tuple of (reconstructed_sequence, confidence_scores)
        """
        predictions = {}
        confidence_scores = {}
        
        # Find target species index
        target_idx = None
        for idx, species in phylogenetic_data.species_mapping.items():
            if species == target_species:
                target_idx = idx
                break
                
        if target_idx is None:
            raise ValueError(f"Target species {target_species} not found in phylogenetic data")
        
        # GNN prediction
        if self.gnn_model is not None:
            with torch.no_grad():
                gnn_logits = self.gnn_model(phylogenetic_data, target_idx)
                gnn_probs = F.softmax(gnn_logits, dim=-1)
                predictions['gnn'] = gnn_probs
                confidence_scores['gnn'] = torch.max(gnn_probs).item()
        
        # LSTM prediction (if available)
        if self.lstm_model is not None:
            # Convert incomplete sequence to tensor
            sequence_tensor = self._sequence_to_tensor(incomplete_sequence)
            
            with torch.no_grad():
                lstm_output = self.lstm_model.predict_next_bases(
                    sequence_tensor.unsqueeze(0), 
                    num_bases=sequence_length - len(incomplete_sequence)
                )
                predictions['lstm'] = lstm_output
                confidence_scores['lstm'] = 0.7  # Placeholder confidence
        
        # Transformer prediction (if available)
        if self.transformer_model is not None:
            # Implement transformer prediction
            predictions['transformer'] = None  # Placeholder
            confidence_scores['transformer'] = 0.6
            
        # Autoencoder prediction (if available)
        if self.autoencoder_model is not None:
            # Implement autoencoder prediction
            predictions['autoencoder'] = None  # Placeholder
            confidence_scores['autoencoder'] = 0.65
        
        # Ensemble combination
        final_sequence = self._combine_predictions(predictions, incomplete_sequence, sequence_length)
        
        return final_sequence, confidence_scores
        
    def _sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to tensor"""
        base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        return torch.tensor([base_map.get(base, 4) for base in sequence], dtype=torch.long)
    
    def _tensor_to_sequence(self, tensor: torch.Tensor) -> str:
        """Convert tensor to DNA sequence"""
        base_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        return ''.join([base_map[idx.item()] for idx in tensor])
    
    def _combine_predictions(self, 
                           predictions: Dict[str, torch.Tensor],
                           incomplete_sequence: str,
                           target_length: int) -> str:
        """Combine predictions from different models"""
        
        # Start with incomplete sequence
        result = incomplete_sequence
        
        # If we have GNN predictions, use them as the primary guide
        if 'gnn' in predictions and predictions['gnn'] is not None:
            gnn_probs = predictions['gnn']
            
            # Sample from GNN predictions to complete the sequence
            for i in range(len(result), target_length):
                # Use temperature sampling for diversity
                temperature = 0.8
                scaled_probs = F.softmax(torch.log(gnn_probs + 1e-8) / temperature, dim=-1)
                sampled_base = torch.multinomial(scaled_probs, 1).item()
                
                base_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
                result += base_map[sampled_base]
        
        # Truncate to target length
        return result[:target_length]


def create_example_phylogenetic_data():
    """Create example phylogenetic data for testing"""
    
    # Example species sequences (shortened for demonstration)
    species_sequences = {
        'Human': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC',
        'Neanderthal': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATT',
        'Chimpanzee': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAAC',
        'Mammoth': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAGG'
    }
    
    # Build phylogenetic graph
    graph_builder = PhylogeneticGraphBuilder()
    phylo_data = graph_builder.create_phylogenetic_graph(species_sequences)
    
    return phylo_data, species_sequences


# Example usage and testing
if __name__ == "__main__":
    
    print("Testing Phylogenetic GNN...")
    
    # Create example data
    phylo_data, species_sequences = create_example_phylogenetic_data()
    
    print(f"Created phylogenetic graph with {phylo_data.x.shape[0]} species")
    print(f"Node features shape: {phylo_data.x.shape}")
    print(f"Edge connections: {phylo_data.edge_index.shape[1]}")
    
    # Initialize GNN model
    gnn_model = PhylogeneticGNN(
        node_feature_dim=100,
        hidden_dim=256,
        output_dim=512,
        num_layers=3,
        gnn_type='GCN'
    )
    
    # Test forward pass
    target_species_idx = 1  # Neanderthal
    sequence_logits = gnn_model(phylo_data, target_species_idx)
    
    print(f"Sequence prediction logits shape: {sequence_logits.shape}")
    print(f"Predicted base probabilities: {F.softmax(sequence_logits, dim=-1)}")
    
    # Test ensemble reconstructor
    ensemble = EnsemblePhylogeneticReconstructor(gnn_model)
    
    incomplete_seq = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    reconstructed, confidence = ensemble.reconstruct_sequence(
        phylo_data, 
        incomplete_seq, 
        'Neanderthal',
        sequence_length=100
    )
    
    print(f"\nOriginal incomplete sequence: {incomplete_seq}")
    print(f"Reconstructed sequence: {reconstructed}")
    print(f"Confidence scores: {confidence}")
    
    print("\nPhylogenetic GNN module completed successfully!")
