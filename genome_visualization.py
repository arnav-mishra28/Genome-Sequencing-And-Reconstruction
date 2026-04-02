"""
Genome Reconstruction Visualization Module
Advanced visualization tools for genomic analysis results

This module provides comprehensive visualization capabilities including:
- Sequence alignment plots
- Mutation distribution heatmaps
- Model performance comparisons
- Phylogenetic relationship graphs
- Confidence score distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class GenomeVisualization:
    """Comprehensive visualization tools for genome reconstruction analysis"""
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.colors = {
            'A': '#FF6B6B',  # Red
            'T': '#4ECDC4',  # Teal
            'G': '#45B7D1',  # Blue
            'C': '#96CEB4',  # Green
            'N': '#FECA57'   # Yellow
        }
    
    def plot_sequence_alignment(self, sequences_dict, title="Sequence Alignment", figsize=(15, 8)):
        """Create visual alignment of multiple DNA sequences"""
        fig, ax = plt.subplots(figsize=figsize)
        
        species_names = list(sequences_dict.keys())
        max_length = max(len(seq) for seq in sequences_dict.values())
        
        # Create matrix for visualization
        for i, (species, sequence) in enumerate(sequences_dict.items()):
            y_pos = len(species_names) - i - 1
            
            for j, base in enumerate(sequence):
                color = self.colors.get(base, '#CCCCCC')
                rect = Rectangle((j, y_pos), 1, 0.8, facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
        
        ax.set_xlim(0, max_length)
        ax.set_ylim(-0.5, len(species_names))
        ax.set_yticks(range(len(species_names)))
        ax.set_yticklabels([name.replace('_', ' ').title() for name in reversed(species_names)])
        ax.set_xlabel('Base Position')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, label=base) 
                          for base, color in self.colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        return fig
    
    def plot_mutation_heatmap(self, mutations_data, figsize=(12, 8)):
        """Create heatmap showing mutation patterns across positions"""
        # Prepare data for heatmap
        species = []
        positions = []
        mutation_types = []
        
        for species_name, mutations in mutations_data.items():
            for mut in mutations:
                species.append(species_name.replace('_', ' ').title())
                positions.append(mut['position'])
                mutation_types.append(mut['type'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Species': species,
            'Position': positions,
            'Mutation_Type': mutation_types
        })
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No mutations detected', ha='center', va='center', fontsize=16)
            ax.set_title('Mutation Analysis', fontsize=14, fontweight='bold')
            return fig
        
        # Create pivot table for heatmap
        pivot_table = df.pivot_table(
            index='Species', 
            columns='Position', 
            values='Mutation_Type', 
            aggfunc='count',
            fill_value=0
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'Number of Mutations'})
        ax.set_title('Mutation Distribution Across Species and Positions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Base Position')
        ax.set_ylabel('Species')
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, training_history, figsize=(15, 5)):
        """Plot training performance for all models"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # LSTM performance
        axes[0].plot(training_history['lstm_losses'], 'b-', linewidth=2, label='LSTM Loss')
        axes[0].set_title('LSTM Training Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Transformer performance
        axes[1].plot(training_history['transformer_losses'], 'r-', linewidth=2, label='Transformer Loss')
        axes[1].set_title('Transformer Training Loss', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Denoiser performance
        axes[2].plot(training_history['denoiser_losses'], 'g-', linewidth=2, label='Denoiser Loss')
        axes[2].set_title('Denoising Autoencoder Loss', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_scores(self, results, figsize=(12, 6)):
        """Plot confidence scores comparison across models and species"""
        # Prepare data
        species_names = []
        models = []
        scores = []
        
        for species, result in results.items():
            confidences = result['model_performance']['individual_confidences']
            for model, score in confidences.items():
                species_names.append(species.replace('_', ' ').title())
                models.append(model.upper())
                scores.append(score)
        
        df = pd.DataFrame({
            'Species': species_names,
            'Model': models,
            'Confidence': scores
        })
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        sns.barplot(data=df, x='Species', y='Confidence', hue='Model', ax=ax1)
        ax1.set_title('Model Confidence Scores by Species', fontweight='bold')
        ax1.set_ylabel('Confidence Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Box plot
        sns.boxplot(data=df, x='Model', y='Confidence', ax=ax2)
        ax2.set_title('Confidence Score Distribution by Model', fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        
        plt.tight_layout()
        return fig
    
    def plot_mutation_effects_pie(self, results, figsize=(10, 8)):
        """Create pie chart showing distribution of mutation effects"""
        # Aggregate all mutation effects
        all_effects = {}
        
        for species, result in results.items():
            effects = result['mutation_analysis']['predicted_effects']
            for effect, count in effects.items():
                if effect in all_effects:
                    all_effects[effect] += count
                else:
                    all_effects[effect] = count
        
        if not all_effects:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No mutation effects to display', ha='center', va='center', fontsize=16)
            ax.set_title('Mutation Effects Distribution', fontsize=14, fontweight='bold')
            return fig
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = []
        sizes = []
        for effect, count in all_effects.items():
            # Simplify long labels
            simplified_label = effect.split('-')[0].replace('_', ' ').title()
            labels.append(simplified_label)
            sizes.append(count)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90
        )
        
        ax.set_title('Distribution of Predicted Mutation Effects', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_genome_browser(self, sequences_dict, mutations_data):
        """Create interactive genome browser using Plotly"""
        fig = make_subplots(
            rows=len(sequences_dict),
            cols=1,
            subplot_titles=[name.replace('_', ' ').title() for name in sequences_dict.keys()],
            vertical_spacing=0.05
        )
        
        for i, (species, sequence) in enumerate(sequences_dict.items()):
            row = i + 1
            
            # Create base sequence visualization
            x_positions = list(range(len(sequence)))
            y_values = [1] * len(sequence)
            colors = [self.colors.get(base, '#CCCCCC') for base in sequence]
            
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Position: {pos}<br>Base: {base}" for pos, base in enumerate(sequence)],
                    hovertemplate='%{text}<extra></extra>',
                    name=species.replace('_', ' ').title(),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Highlight mutations
            if species in mutations_data:
                mut_positions = [mut['position'] for mut in mutations_data[species]]
                mut_y = [1.2] * len(mut_positions)
                
                fig.add_trace(
                    go.Scatter(
                        x=mut_positions,
                        y=mut_y,
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='diamond'),
                        name='Mutations',
                        text=[f"Mutation at {pos}" for pos in mut_positions],
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=(i == 0)
                    ),
                    row=row, col=1
                )
            
            # Update y-axis for each subplot
            fig.update_yaxes(
                range=[0.5, 1.5],
                showticklabels=False,
                row=row, col=1
            )
        
        fig.update_layout(
            title='Interactive Genome Browser',
            height=200 * len(sequences_dict),
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Base Position', row=len(sequences_dict), col=1)
        
        return fig
    
    def generate_comprehensive_report(self, results, training_history, output_dir='.'):
        """Generate comprehensive visual report"""
        print("Generating comprehensive visualization report...")
        
        # Extract data for visualization
        sequences = {}
        mutations_by_species = {}
        
        for species, result in results.items():
            # Get ensemble prediction as the main sequence
            sequences[species] = result['reconstructed_sequences'].get('ensemble', '')
            mutations_by_species[species] = result['mutation_analysis']['mutations_by_position']
        
        # Create all visualizations
        figures = {}
        
        # 1. Sequence alignment
        figures['alignment'] = self.plot_sequence_alignment(
            sequences, 
            title="Reconstructed Genome Sequences Alignment"
        )
        
        # 2. Mutation heatmap
        figures['mutations'] = self.plot_mutation_heatmap(mutations_by_species)
        
        # 3. Model performance
        figures['performance'] = self.plot_model_performance(training_history)
        
        # 4. Confidence scores
        figures['confidence'] = self.plot_confidence_scores(results)
        
        # 5. Mutation effects
        figures['effects'] = self.plot_mutation_effects_pie(results)
        
        # Save all figures
        for name, fig in figures.items():
            fig.savefig(f'{output_dir}/genome_{name}_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved {name} analysis plot")
        
        # Create interactive browser
        interactive_fig = self.create_interactive_genome_browser(sequences, mutations_by_species)
        interactive_fig.write_html(f'{output_dir}/interactive_genome_browser.html')
        print("Saved interactive genome browser")
        
        plt.close('all')  # Clean up
        
        return figures

def create_summary_statistics(results):
    """Generate summary statistics for the analysis"""
    summary = {
        'total_species': len(results),
        'total_mutations': 0,
        'average_confidence': 0.0,
        'mutation_types_summary': {},
        'species_statistics': {}
    }
    
    confidence_scores = []
    
    for species, result in results.items():
        # Species-specific statistics
        stats = result['sequence_statistics']
        mutations = result['mutation_analysis']['mutations_by_position']
        confidence = stats['confidence_score']
        
        summary['species_statistics'][species] = {
            'sequence_length': stats['reconstructed_length'],
            'mutation_count': len(mutations),
            'confidence_score': confidence
        }
        
        # Aggregate statistics
        summary['total_mutations'] += len(mutations)
        confidence_scores.append(confidence)
        
        # Mutation types
        for mutation in mutations:
            mut_type = mutation['type']
            if mut_type in summary['mutation_types_summary']:
                summary['mutation_types_summary'][mut_type] += 1
            else:
                summary['mutation_types_summary'][mut_type] = 1
    
    summary['average_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
    
    return summary

if __name__ == "__main__":
    print("Genome Visualization Module loaded successfully")
