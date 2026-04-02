"""
Graph Neural Network for Phylogenetic Analysis
==============================================

GNN model that incorporates evolutionary relationships between species
to improve genome reconstruction accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
import networkx as nx
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class PhylogeneticGraph:
    """Build phylogenetic graphs for species relationships"""
    
    def __init__(self):
        self.species_info = {
            'human': {
                'scientific_name': 'Homo sapiens',
                'branch_length': 0.0,  # Reference species
                'features': [1, 0, 0, 0, 0]  # [mammal, primate, hominid, extinct, large_genome]
            },
            'neanderthal': {
                'scientific_name': 'Homo neanderthalensis',
                'branch_length': 0.5,  # Million years ago
                'features': [1, 0, 1, 1, 0]
            },
            'elephant': {
                'scientific_name': 'Loxodonta africana',
                'branch_length': 95.0,  # Diverged ~95 MYA
                'features': [1, 0, 0, 0, 1]
            },
            'mammoth': {
                'scientific_name': 'Mammuthus primigenius',
                'branch_length': 6.0,  # Diverged from elephants ~6 MYA
                'features': [1, 0, 0, 1, 1]
            },
            'chimpanzee': {
                'scientific_name': 'Pan troglodytes',
                'branch_length': 7.0,  # Diverged ~7 MYA
                'features': [1, 1, 0, 0, 0]
            }
        }
    
    def build_phylogenetic_tree(self):
        """Build phylogenetic tree as a graph"""
        # Create networkx graph
        G = nx.Graph()
        
        # Add species nodes
        for species, info in self.species_info.items():
            G.add_node(species, **info)
        
        # Add evolutionary relationships (edges)
        relationships = [
            ('human', 'neanderthal', 0.5),     # Close relationship
            ('human', 'chimpanzee', 7.0),      # Common ancestor
            ('elephant', 'mammoth', 6.0),      # Very close
            ('human', 'elephant', 95.0),       # Distant mammalian relationship
            ('neanderthal', 'chimpanzee', 7.5), # Through human lineage
            ('chimpanzee', 'elephant', 95.0),   # Mammalian
            ('mammoth', 'human', 95.0),        # Through elephant
            ('mammoth', 'chimpanzee', 95.0),   # Mammalian
            ('neanderthal', 'elephant', 95.0), # Mammalian
            ('neanderthal', 'mammoth', 95.0)   # Mammalian
        ]
        
        for species1, species2, distance in relationships:
            # Edge weight is inverse of evolutionary distance
            weight = 1.0 / (1.0 + distance)
            G.add_edge(species1, species2, weight=weight, distance=distance)
        
        return G
    
    def graph_to_pyg_data(self, G, sequence_features=None):
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        
        # Create node mapping
        nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        
        # Node features
        node_features = []
        for node in nodes:
            features = self.species_info[node]['features']
            if sequence_features and node in sequence_features:
                # Add sequence-specific features
                features = features + sequence_features[node]
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Edge indices and weights
        edge_index = []
        edge_attr = []
        
        for edge in G.edges(data=True):
            src, dst, data = edge
            src_idx = node_mapping[src]
            dst_idx = node_mapping[dst]
            
            # Add both directions (undirected graph)
            edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])
            weight = data['weight']
            edge_attr.extend([weight, weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr), node_mapping

class PhylogeneticGNN(pl.LightningModule):
    """Graph Neural Network for phylogenetic analysis"""
    
    def __init__(self, node_features=5, hidden_dim=256, num_layers=3, 
                 output_dim=128, dropout=0.2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=4, dropout=dropout, concat=False))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False))
        
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout, concat=False))
        
        # Transformer layers for sequence integration
        self.transformer_conv = TransformerConv(
            output_dim, output_dim, heads=8, dropout=dropout
        )
        
        # Prediction heads
        self.sequence_predictor = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 5)  # A, T, G, C, N
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.learning_rate = learning_rate
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Graph convolutions with residual connections
        for i, conv in enumerate(self.convs):
            if i == 0:
                x = F.relu(conv(x, edge_index))
            else:
                # Residual connection
                x_new = F.relu(conv(x, edge_index))
                if x_new.size() == x.size():
                    x = x + x_new
                else:
                    x = x_new
        
        # Transformer convolution for long-range dependencies
        x = self.transformer_conv(x, edge_index)
        
        return x
    
    def predict_from_relatives(self, target_species, relative_sequences, phylo_graph):
        """Predict sequence for target species using related species"""
        
        # Convert graph to PyG format
        pyg_data, node_mapping = phylo_graph.graph_to_pyg_data(
            phylo_graph.build_phylogenetic_tree(),
            sequence_features=self._extract_sequence_features(relative_sequences)
        )
        
        # Forward pass
        node_embeddings = self(pyg_data)
        
        # Get target species embedding
        target_idx = node_mapping[target_species]
        target_embedding = node_embeddings[target_idx]
        
        # Predict sequence properties
        sequence_logits = self.sequence_predictor(target_embedding)
        confidence = self.confidence_predictor(target_embedding)
        
        return {
            'embedding': target_embedding,
            'sequence_logits': sequence_logits,
            'confidence': confidence,
            'node_embeddings': node_embeddings
        }
    
    def _extract_sequence_features(self, sequences):
        """Extract features from DNA sequences"""
        features = {}
        
        for species, sequence in sequences.items():
            if isinstance(sequence, str):
                seq_features = [
                    len(sequence),
                    sequence.count('G') + sequence.count('C'),  # GC count
                    sequence.count('N'),  # Missing bases
                    sequence.count('A'),
                    sequence.count('T')
                ]
                # Normalize
                total_bases = len(sequence)
                if total_bases > 0:
                    seq_features = [f / total_bases for f in seq_features]
                features[species] = seq_features
        
        return features
    
    def training_step(self, batch, batch_idx):
        # This would be implemented based on specific training objectives
        # For now, return a placeholder
        return torch.tensor(0.0, requires_grad=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class HybridPhyloGenomeModel(pl.LightningModule):
    """Hybrid model combining GNN with sequence models"""
    
    def __init__(self, sequence_model, phylo_gnn, fusion_dim=512):
        super().__init__()
        
        self.sequence_model = sequence_model
        self.phylo_gnn = phylo_gnn
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(sequence_model.hparams.hidden_size + phylo_gnn.hparams.output_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 5)  # Final sequence prediction
        )
        
        self.phylo_weight = nn.Parameter(torch.tensor(0.5))  # Learnable fusion weight
    
    def forward(self, sequence_input, phylo_data, species_id):
        # Get sequence features
        seq_features = self.sequence_model.forward(sequence_input, return_confidence=False)
        
        # Get phylogenetic features
        phylo_embeddings = self.phylo_gnn(phylo_data)
        phylo_features = phylo_embeddings[species_id].unsqueeze(0).expand(
            seq_features.size(0), seq_features.size(1), -1
        )
        
        # Fusion
        combined_features = torch.cat([seq_features, phylo_features], dim=-1)
        output = self.fusion(combined_features)
        
        # Weighted combination
        weight = torch.sigmoid(self.phylo_weight)
        final_output = weight * output + (1 - weight) * seq_features
        
        return final_output
    
    def predict_extinct_sequence(self, incomplete_sequence, related_species_data, target_species):
        """Predict complete sequence for extinct species"""
        
        # Create phylogenetic graph
        phylo_graph = PhylogeneticGraph()
        graph = phylo_graph.build_phylogenetic_tree()
        
        # Get related sequences
        related_sequences = {species: data['sequence'] for species, data in related_species_data.items()}
        
        # Convert to PyG data
        pyg_data, node_mapping = phylo_graph.graph_to_pyg_data(graph, related_sequences)
        
        # Get species ID
        species_id = node_mapping[target_species]
        
        # Encode incomplete sequence
        sequence_input = self._encode_sequence(incomplete_sequence)
        
        # Forward pass
        predictions = self.forward(sequence_input, pyg_data, species_id)
        
        # Decode predictions
        predicted_sequence = self._decode_sequence(predictions, incomplete_sequence)
        
        return predicted_sequence
    
    def _encode_sequence(self, sequence):
        """Encode DNA sequence for model input"""
        encoded = torch.tensor([DNA_TO_INT.get(base, DNA_TO_INT['N']) 
                               for base in sequence.upper()], dtype=torch.long)
        return encoded.unsqueeze(0)
    
    def _decode_sequence(self, predictions, original_sequence):
        """Decode model predictions to DNA sequence"""
        predicted_indices = torch.argmax(predictions, dim=-1).squeeze(0)
        
        # Fill in missing bases
        completed_sequence = ""
        for i, (original_base, pred_idx) in enumerate(zip(original_sequence, predicted_indices)):
            if original_base == 'N':
                completed_sequence += INT_TO_DNA[pred_idx.item()]
            else:
                completed_sequence += original_base
        
        return completed_sequence

def create_species_network():
    """Create a comprehensive species network with more organisms"""
    
    extended_species = {
        'human': {'features': [1, 1, 1, 0, 0], 'genome_size': 3200},
        'neanderthal': {'features': [1, 1, 1, 1, 0], 'genome_size': 3200},
        'denisovan': {'features': [1, 1, 1, 1, 0], 'genome_size': 3200},
        'chimpanzee': {'features': [1, 1, 0, 0, 0], 'genome_size': 3100},
        'gorilla': {'features': [1, 1, 0, 0, 0], 'genome_size': 3100},
        'elephant': {'features': [1, 0, 0, 0, 1], 'genome_size': 3400},
        'mammoth': {'features': [1, 0, 0, 1, 1], 'genome_size': 3400},
        'mastodon': {'features': [1, 0, 0, 1, 1], 'genome_size': 3400},
        'mouse': {'features': [1, 0, 0, 0, 0], 'genome_size': 2700},
        'rat': {'features': [1, 0, 0, 0, 0], 'genome_size': 2900},
    }
    
    # Define evolutionary distances (in millions of years)
    distances = {
        ('human', 'neanderthal'): 0.5,
        ('human', 'denisovan'): 0.6,
        ('human', 'chimpanzee'): 7.0,
        ('human', 'gorilla'): 10.0,
        ('elephant', 'mammoth'): 6.0,
        ('elephant', 'mastodon'): 25.0,
        ('mouse', 'rat'): 15.0,
        # Cross-order relationships
        ('human', 'elephant'): 95.0,
        ('human', 'mouse'): 95.0,
        ('elephant', 'mouse'): 95.0,
    }
    
    return extended_species, distances

if __name__ == "__main__":
    # Test the phylogenetic GNN
    phylo_graph = PhylogeneticGraph()
    graph = phylo_graph.build_phylogenetic_tree()
    
    # Convert to PyG format
    pyg_data, node_mapping = phylo_graph.graph_to_pyg_data(graph)
    
    # Initialize model
    model = PhylogeneticGNN(node_features=pyg_data.x.size(1))
    
    # Test forward pass
    with torch.no_grad():
        output = model(pyg_data)
    
    print(f"Graph nodes: {len(graph.nodes())}")
    print(f"Graph edges: {len(graph.edges())}")
    print(f"Node mapping: {node_mapping}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
