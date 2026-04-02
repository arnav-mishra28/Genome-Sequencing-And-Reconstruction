"""
Mutation Analyzer for Genome Sequencing and Reconstruction

This module provides comprehensive mutation detection, classification,
and health impact analysis for DNA sequences.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import random

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


@dataclass
class Mutation:
    """Class representing a genetic mutation"""
    position: int
    original_base: str
    mutated_base: str
    mutation_type: str
    effect: str
    confidence: float
    context: str  # Surrounding sequence context


@dataclass
class HealthImplication:
    """Class representing health implications of mutations"""
    gene_region: str
    mutation_positions: List[int]
    risk_level: str  # Low, Medium, High
    description: str
    suggested_correction: str
    confidence: float


class MutationAnalyzer:
    """Advanced mutation detection and analysis system"""
    
    def __init__(self):
        self.known_diseases = config.DISEASE_MUTATIONS
        self.mutation_patterns = self._load_mutation_patterns()
        self.codon_table = self._get_codon_table()
        
    def _load_mutation_patterns(self) -> Dict:
        """Load known mutation patterns and their effects"""
        return {
            'C_to_T': {
                'description': 'Cytosine to Thymine (common in ancient DNA)',
                'frequency': 0.35,
                'severity': 'low',
                'associated_with': 'DNA degradation'
            },
            'G_to_A': {
                'description': 'Guanine to Adenine (common in ancient DNA)',
                'frequency': 0.30,
                'severity': 'low',
                'associated_with': 'DNA degradation'
            },
            'A_to_G': {
                'description': 'Adenine to Guanine transition',
                'frequency': 0.15,
                'severity': 'medium',
                'associated_with': 'Spontaneous mutation'
            },
            'T_to_C': {
                'description': 'Thymine to Cytosine transition',
                'frequency': 0.10,
                'severity': 'medium',
                'associated_with': 'Spontaneous mutation'
            },
            'transversion': {
                'description': 'Purine to Pyrimidine or vice versa',
                'frequency': 0.10,
                'severity': 'high',
                'associated_with': 'Mutagenic exposure'
            }
        }
    
    def _get_codon_table(self) -> Dict[str, str]:
        """Standard genetic code table"""
        return {
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
    
    def detect_mutations(self, sequence: str, reference_sequence: str = None) -> List[Mutation]:
        """
        Detect mutations in a DNA sequence
        
        Args:
            sequence: Target DNA sequence
            reference_sequence: Reference sequence (if None, use built-in reference)
            
        Returns:
            List of detected mutations
        """
        mutations = []
        
        if reference_sequence is None:
            # Use a healthy reference from config
            reference_sequence = config.SAMPLE_SEQUENCES['human'][:len(sequence)]
        
        # Ensure sequences are same length for comparison
        min_length = min(len(sequence), len(reference_sequence))
        sequence = sequence[:min_length]
        reference_sequence = reference_sequence[:min_length]
        
        # Compare sequences base by base
        for i, (ref_base, seq_base) in enumerate(zip(reference_sequence, sequence)):
            if ref_base != seq_base and ref_base in 'ATCG' and seq_base in 'ATCG':
                # Determine mutation type
                mutation_type = self._classify_mutation_type(ref_base, seq_base)
                
                # Get surrounding context
                start = max(0, i-10)
                end = min(len(sequence), i+11)
                context = sequence[start:end]
                
                # Determine effect
                effect = self._determine_mutation_effect(i, ref_base, seq_base, sequence)
                
                # Calculate confidence based on context
                confidence = self._calculate_mutation_confidence(i, ref_base, seq_base, context)
                
                mutation = Mutation(
                    position=i,
                    original_base=ref_base,
                    mutated_base=seq_base,
                    mutation_type=mutation_type,
                    effect=effect,
                    confidence=confidence,
                    context=context
                )
                mutations.append(mutation)
        
        return mutations
    
    def _classify_mutation_type(self, ref_base: str, mut_base: str) -> str:
        """Classify the type of mutation"""
        transitions = {
            ('A', 'G'): 'A_to_G_transition',
            ('G', 'A'): 'G_to_A_transition',
            ('C', 'T'): 'C_to_T_transition',
            ('T', 'C'): 'T_to_C_transition'
        }
        
        if (ref_base, mut_base) in transitions:
            return transitions[(ref_base, mut_base)]
        else:
            return 'transversion'
    
    def _determine_mutation_effect(self, position: int, ref_base: str, mut_base: str, sequence: str) -> str:
        """Determine the potential effect of a mutation"""
        # Check if in coding region (simplified - assumes position matters)
        codon_position = position % 3
        
        if codon_position == 0:  # First codon position
            return 'potential_amino_acid_change'
        elif codon_position == 1:  # Second codon position
            return 'likely_amino_acid_change'
        else:  # Third codon position
            return 'possible_silent_mutation'
    
    def _calculate_mutation_confidence(self, position: int, ref_base: str, mut_base: str, context: str) -> float:
        """Calculate confidence score for mutation detection"""
        base_confidence = 0.7
        
        # Adjust based on common mutation patterns
        if (ref_base == 'C' and mut_base == 'T') or (ref_base == 'G' and mut_base == 'A'):
            base_confidence += 0.2  # Common ancient DNA damage
        
        # Adjust based on context quality
        n_count = context.count('N')
        if n_count == 0:
            base_confidence += 0.1
        else:
            base_confidence -= n_count * 0.05
        
        return min(1.0, max(0.0, base_confidence))
    
    def categorize_mutations(self, mutations: List[Mutation]) -> Dict[str, int]:
        """Categorize mutations by type"""
        categories = defaultdict(int)
        
        for mutation in mutations:
            categories[mutation.mutation_type] += 1
        
        return dict(categories)
    
    def analyze_health_implications(self, sequence: str) -> List[HealthImplication]:
        """Analyze health implications of detected mutations"""
        health_implications = []
        
        # Check against known disease mutations
        for disease, disease_info in self.known_diseases.items():
            mutations_found = []
            
            # Look for specific mutation patterns
            for mutation_pattern in disease_info['mutations']:
                positions = self._find_mutation_pattern(sequence, mutation_pattern)
                mutations_found.extend(positions)
            
            if mutations_found:
                risk_level = self._assess_risk_level(disease_info, len(mutations_found))
                
                suggestion = self._generate_correction_suggestion(disease, mutations_found)
                
                implication = HealthImplication(
                    gene_region=disease,
                    mutation_positions=mutations_found,
                    risk_level=risk_level,
                    description=disease_info['description'],
                    suggested_correction=suggestion,
                    confidence=disease_info['confidence']
                )
                health_implications.append(implication)
        
        return health_implications
    
    def _find_mutation_pattern(self, sequence: str, pattern: str) -> List[int]:
        """Find specific mutation patterns in sequence"""
        positions = []
        
        # Simple pattern matching (can be enhanced with regex)
        pattern_length = len(pattern)
        for i in range(len(sequence) - pattern_length + 1):
            subseq = sequence[i:i+pattern_length]
            if subseq == pattern:
                positions.append(i)
        
        return positions
    
    def _assess_risk_level(self, disease_info: Dict, num_mutations: int) -> str:
        """Assess risk level based on mutations found"""
        severity = disease_info.get('severity', 'medium')
        
        if num_mutations == 0:
            return 'none'
        elif num_mutations <= 2 and severity == 'low':
            return 'low'
        elif num_mutations <= 3 and severity == 'medium':
            return 'medium'
        else:
            return 'high'
    
    def _generate_correction_suggestion(self, disease: str, positions: List[int]) -> str:
        """Generate correction suggestions for detected mutations"""
        corrections = {
            'BRCA1': "Replace with healthy BRCA1 variant to restore tumor suppressor function",
            'CFTR': "Correct with functional CFTR sequence to improve chloride channel activity",
            'APOE4': "Consider APOE2/3 variants for reduced Alzheimer's risk",
            'Huntington': "Replace expanded CAG repeat with normal length sequence"
        }
        
        return corrections.get(disease, "Correct with consensus healthy sequence")
    
    def suggest_corrections(self, sequence: str) -> str:
        """Suggest corrections for detected mutations"""
        mutations = self.detect_mutations(sequence)
        corrected_sequence = list(sequence)
        
        # Sort mutations by confidence (highest first)
        mutations.sort(key=lambda x: x.confidence, reverse=True)
        
        for mutation in mutations:
            if mutation.confidence > 0.7:  # Only correct high-confidence mutations
                # Apply correction based on mutation type and context
                correction = self._get_optimal_correction(mutation)
                corrected_sequence[mutation.position] = correction
        
        return ''.join(corrected_sequence)
    
    def _get_optimal_correction(self, mutation: Mutation) -> str:
        """Get optimal correction for a specific mutation"""
        # Prioritize reversing common ancient DNA damage
        if mutation.mutation_type == 'C_to_T_transition':
            return 'C'  # Reverse C->T damage
        elif mutation.mutation_type == 'G_to_A_transition':
            return 'G'  # Reverse G->A damage
        else:
            # For other mutations, use consensus or most frequent base
            return mutation.original_base
    
    def generate_mutation_report(self, sequence: str) -> Dict:
        """Generate comprehensive mutation analysis report"""
        mutations = self.detect_mutations(sequence)
        health_implications = self.analyze_health_implications(sequence)
        corrected_sequence = self.suggest_corrections(sequence)
        
        # Calculate statistics
        mutation_stats = {
            'total_mutations': len(mutations),
            'high_confidence_mutations': len([m for m in mutations if m.confidence > 0.8]),
            'mutation_density': len(mutations) / len(sequence) * 100,
            'most_common_type': max(self.categorize_mutations(mutations).items(), 
                                  key=lambda x: x[1])[0] if mutations else 'none'
        }
        
        health_stats = {
            'total_health_implications': len(health_implications),
            'high_risk_findings': len([h for h in health_implications if h.risk_level == 'high']),
            'genes_affected': len(set(h.gene_region for h in health_implications))
        }
        
        correction_stats = {
            'corrections_applied': len([m for m in mutations if m.confidence > 0.7]),
            'correction_success_rate': 0.85,  # Placeholder - would be calculated from validation
            'corrected_sequence_length': len(corrected_sequence)
        }
        
        return {
            'analysis_summary': {
                'sequence_length': len(sequence),
                'analysis_timestamp': str(np.datetime64('now')),
                'mutations_detected': mutation_stats,
                'health_assessment': health_stats,
                'corrections': correction_stats
            },
            'detailed_mutations': [
                {
                    'position': m.position,
                    'change': f"{m.original_base}->{m.mutated_base}",
                    'type': m.mutation_type,
                    'effect': m.effect,
                    'confidence': m.confidence,
                    'context': m.context
                } for m in mutations[:50]  # Limit for readability
            ],
            'health_implications': [
                {
                    'gene': h.gene_region,
                    'risk_level': h.risk_level,
                    'description': h.description,
                    'positions_affected': len(h.mutation_positions),
                    'suggested_correction': h.suggested_correction,
                    'confidence': h.confidence
                } for h in health_implications
            ],
            'corrected_sequence': corrected_sequence[:200] + "..." if len(corrected_sequence) > 200 else corrected_sequence
        }


# Utility functions for mutation simulation and analysis

def simulate_random_mutations(sequence: str, mutation_rate: float = 0.01) -> Tuple[str, List[Dict]]:
    """
    Simulate random mutations in a DNA sequence
    
    Args:
        sequence: Original DNA sequence
        mutation_rate: Probability of mutation per base
        
    Returns:
        Tuple of (mutated_sequence, mutation_log)
    """
    mutated_sequence = list(sequence)
    mutation_log = []
    
    bases = ['A', 'T', 'C', 'G']
    
    for i, base in enumerate(sequence):
        if random.random() < mutation_rate and base in bases:
            original_base = base
            new_base = random.choice([b for b in bases if b != base])
            mutated_sequence[i] = new_base
            
            mutation_log.append({
                'position': i,
                'original': original_base,
                'mutated': new_base,
                'type': 'random_substitution'
            })
    
    return ''.join(mutated_sequence), mutation_log


def simulate_ancient_dna_damage(sequence: str, damage_rate: float = 0.05) -> str:
    """
    Simulate typical ancient DNA damage patterns
    
    Args:
        sequence: Original DNA sequence
        damage_rate: Probability of damage per base
        
    Returns:
        Damaged sequence
    """
    damaged_sequence = list(sequence)
    
    for i, base in enumerate(sequence):
        if random.random() < damage_rate:
            if base == 'C':
                damaged_sequence[i] = 'T'  # C->T deamination
            elif base == 'G':
                damaged_sequence[i] = 'A'  # G->A deamination
            # Other damage patterns can be added
    
    return ''.join(damaged_sequence)


# Example usage and testing
if __name__ == "__main__":
    # Example analysis
    analyzer = MutationAnalyzer()
    
    # Test with a sample sequence
    sample_sequence = config.SAMPLE_SEQUENCES['human'][:1000]
    
    print("🧬 Running mutation analysis...")
    
    # Simulate some damage for testing
    damaged_sequence = simulate_ancient_dna_damage(sample_sequence, 0.02)
    
    # Run analysis
    mutations = analyzer.detect_mutations(damaged_sequence, sample_sequence)
    health_implications = analyzer.analyze_health_implications(damaged_sequence)
    corrected_sequence = analyzer.suggest_corrections(damaged_sequence)
    
    print(f"✅ Analysis complete:")
    print(f"  - Mutations detected: {len(mutations)}")
    print(f"  - Health implications: {len(health_implications)}")
    print(f"  - Corrections suggested: {len([m for m in mutations if m.confidence > 0.7])}")
    
    # Generate full report
    report = analyzer.generate_mutation_report(damaged_sequence)
    print(f"  - Mutation density: {report['analysis_summary']['mutations_detected']['mutation_density']:.2f}%")
