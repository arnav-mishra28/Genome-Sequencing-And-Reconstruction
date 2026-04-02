#!/usr/bin/env python3
"""
Main Execution Script for Genome Sequencing and Reconstruction System

This script orchestrates the entire genome reconstruction pipeline:
1. Data loading and preprocessing
2. Model training (LSTM, Transformer, Autoencoder, GNN)
3. Sequence alignment and phylogenetic analysis
4. Mutation analysis and reconstruction
5. Comprehensive reporting and visualization

Usage: python run_genome_analysis.py
"""

import sys
import os
import json
import pickle
import datetime
import traceback
from pathlib import Path

# Import our modules
from genome_reconstruction_system import run_genome_reconstruction_analysis, HybridGenomeReconstructor
from genome_visualization import GenomeVisualization, create_summary_statistics
from advanced_alignment import run_advanced_alignment_analysis

def setup_output_directory():
    """Create timestamped output directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"genome_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results(results, output_dir):
    """Save all analysis results"""
    # Save main results as JSON
    json_results = {}
    for species, result in results.items():
        # Convert to JSON-serializable format
        json_result = {
            'species': result['species'],
            'timestamp': result['timestamp'],
            'sequence_statistics': result['sequence_statistics'],
            'mutation_analysis': {
                'mutation_types': result['mutation_analysis']['mutation_types'],
                'predicted_effects': result['mutation_analysis']['predicted_effects'],
                'mutation_count': len(result['mutation_analysis']['mutations_by_position'])
            },
            'model_performance': result['model_performance'],
            'reconstructed_sequences': result['reconstructed_sequences']
        }
        json_results[species] = json_result
    
    with open(f"{output_dir}/analysis_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save detailed results as pickle for Python analysis
    with open(f"{output_dir}/detailed_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_dir}/")

def generate_text_report(results, alignment_results, summary_stats, output_dir):
    """Generate comprehensive text report"""
    report_path = f"{output_dir}/comprehensive_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENOME SEQUENCING AND RECONSTRUCTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Species Analyzed: {summary_stats['total_species']}\n")
        f.write(f"Total Mutations Detected: {summary_stats['total_mutations']}\n")
        f.write(f"Average Reconstruction Confidence: {summary_stats['average_confidence']:.3f}\n")
        f.write(f"Most Common Mutation Type: {max(summary_stats['mutation_types_summary'], key=summary_stats['mutation_types_summary'].get) if summary_stats['mutation_types_summary'] else 'None'}\n\n")
        
        # Detailed Species Analysis
        f.write("DETAILED SPECIES ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        for species, result in results.items():
            f.write(f"\n{species.upper().replace('_', ' ')}\n")
            f.write("~" * len(species) + "\n")
            
            stats = result['sequence_statistics']
            mutations = result['mutation_analysis']['mutations_by_position']
            
            f.write(f"Sequence Length: {stats['reconstructed_length']} bases\n")
            f.write(f"Mutations Detected: {len(mutations)}\n")
            f.write(f"Reconstruction Confidence: {stats['confidence_score']:.3f}\n")
            
            # Mutation breakdown
            mutation_types = {}
            for mut in mutations:
                mut_type = mut['type']
                mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
            
            if mutation_types:
                f.write("Mutation Types:\n")
                for mut_type, count in mutation_types.items():
                    f.write(f"  - {mut_type.replace('_', ' ').title()}: {count}\n")
            
            # Model performance
            f.write("Model Performance:\n")
            for model, confidence in result['model_performance']['individual_confidences'].items():
                f.write(f"  - {model.upper()}: {confidence:.3f}\n")
            
            f.write(f"Best Model: {result['model_performance']['best_model']}\n")
        
        # Alignment Analysis
        if alignment_results:
            f.write("\nSEQUENCE ALIGNMENT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            quality_stats = alignment_results['quality_statistics']
            f.write(f"Alignment Length: {quality_stats['alignment_length']} bases\n")
            f.write(f"Number of Sequences: {quality_stats['n_sequences']}\n")
            f.write(f"Average Conservation: {quality_stats['average_conservation']:.3f}\n")
            f.write(f"Gap Frequency: {quality_stats['total_gap_frequency']:.3f}\n")
            f.write(f"Alignment Quality Score: {quality_stats['alignment_quality_score']:.3f}\n")
            f.write(f"Highly Conserved Positions: {len(quality_stats['conserved_positions'])}\n")
            f.write(f"High Gap Positions: {len(quality_stats['gap_positions'])}\n")
        
        # Mutation Analysis Summary
        f.write("\nMUTATION ANALYSIS SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        all_effects = {}
        for species, result in results.items():
            effects = result['mutation_analysis']['predicted_effects']
            for effect, count in effects.items():
                all_effects[effect] = all_effects.get(effect, 0) + count
        
        if all_effects:
            f.write("Predicted Mutation Effects:\n")
            for effect, count in sorted(all_effects.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - {effect}: {count}\n")
        
        # Technical Details
        f.write("\nTECHNICAL DETAILS\n")
        f.write("-" * 40 + "\n")
        f.write("Models Used:\n")
        f.write("  - LSTM: Bidirectional LSTM for sequence prediction\n")
        f.write("  - Transformer: DNA-BERT style masked language model\n")
        f.write("  - Autoencoder: Denoising autoencoder for error correction\n")
        f.write("  - GNN: Graph Neural Network for phylogenetic analysis\n")
        f.write("  - Ensemble: Weighted voting combination of all models\n\n")
        
        f.write("Alignment Method: Progressive Multiple Sequence Alignment\n")
        f.write("Distance Calculation: Kimura 2-parameter model\n")
        f.write("Tree Construction: Neighbor-Joining algorithm\n")
        
        # Recommendations
        f.write("\nRECOMMENDations FOR FURTHER ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Validate reconstructions using experimental methods\n")
        f.write("2. Expand training dataset with more ancient DNA samples\n")
        f.write("3. Incorporate additional phylogenetic constraints\n")
        f.write("4. Apply functional annotation to identify important mutations\n")
        f.write("5. Cross-validate with other reconstruction methods\n")
    
    print(f"Comprehensive report saved to {report_path}")

def create_sequence_fasta_files(results, alignment_results, output_dir):
    """Create FASTA files for reconstructed sequences"""
    # Original reconstructed sequences
    fasta_path = f"{output_dir}/reconstructed_sequences.fasta"
    with open(fasta_path, 'w') as f:
        for species, result in results.items():
            sequence = result['reconstructed_sequences'].get('ensemble', '')
            f.write(f">{species}_reconstructed\n")
            f.write(f"{sequence}\n")
    
    # Aligned sequences if available
    if alignment_results and 'aligned_sequences' in alignment_results:
        aligned_fasta_path = f"{output_dir}/aligned_sequences.fasta"
        with open(aligned_fasta_path, 'w') as f:
            for seq_record in alignment_results['aligned_sequences']:
                f.write(f">{seq_record.id}_aligned\n")
                f.write(f"{str(seq_record.seq)}\n")
        print(f"Aligned sequences saved to {aligned_fasta_path}")
    
    print(f"Reconstructed sequences saved to {fasta_path}")

def main():
    """Main execution function"""
    print("🧬 Starting Genome Sequencing and Reconstruction Analysis")
    print("=" * 80)
    
    try:
        # Setup output directory
        output_dir = setup_output_directory()
        print(f"📁 Output directory: {output_dir}")
        
        # Run main genome reconstruction analysis
        print("\n🔬 Running genome reconstruction analysis...")
        results, model, training_history = run_genome_reconstruction_analysis()
        
        # Extract sequences for alignment analysis
        sequences_for_alignment = {}
        for species, result in results.items():
            sequences_for_alignment[species] = result['reconstructed_sequences'].get('ensemble', '')
        
        # Run alignment analysis
        print("\n🧬 Running sequence alignment analysis...")
        try:
            alignment_results = run_advanced_alignment_analysis(sequences_for_alignment)
        except Exception as e:
            print(f"⚠️  Alignment analysis failed: {e}")
            alignment_results = None
        
        # Generate visualizations
        print("\n📊 Creating comprehensive visualizations...")
        visualizer = GenomeVisualization()
        visualization_figures = visualizer.generate_comprehensive_report(
            results, training_history, output_dir
        )
        
        # Generate additional alignment visualizations
        if alignment_results and alignment_results['visualizations']['alignment']:
            alignment_results['visualizations']['alignment'].savefig(
                f"{output_dir}/sequence_alignment_detailed.png", 
                dpi=300, bbox_inches='tight'
            )
            print("💾 Saved detailed alignment visualization")
        
        if alignment_results and alignment_results['visualizations']['tree']:
            alignment_results['visualizations']['tree'].savefig(
                f"{output_dir}/phylogenetic_tree.png", 
                dpi=300, bbox_inches='tight'
            )
            print("🌳 Saved phylogenetic tree")
        
        # Generate summary statistics
        print("\n📈 Generating summary statistics...")
        summary_stats = create_summary_statistics(results)
        
        # Save all results
        print("\n💾 Saving results...")
        save_results(results, output_dir)
        
        # Generate comprehensive text report
        print("\n📝 Generating comprehensive report...")
        generate_text_report(results, alignment_results, summary_stats, output_dir)
        
        # Create FASTA files
        print("\n🧬 Creating FASTA files...")
        create_sequence_fasta_files(results, alignment_results, output_dir)
        
        # Save summary statistics
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Save training history
        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"📁 Results saved in: {output_dir}")
        print(f"🧬 Species analyzed: {summary_stats['total_species']}")
        print(f"🔬 Mutations detected: {summary_stats['total_mutations']}")
        print(f"📊 Average confidence: {summary_stats['average_confidence']:.3f}")
        print("\n📋 Files generated:")
        print("   • analysis_results.json - Main results in JSON format")
        print("   • detailed_results.pkl - Full Python objects for further analysis")
        print("   • comprehensive_report.txt - Human-readable detailed report")
        print("   • reconstructed_sequences.fasta - FASTA file of reconstructed sequences")
        print("   • aligned_sequences.fasta - FASTA file of aligned sequences")
        print("   • summary_statistics.json - Summary statistics")
        print("   • training_history.json - Model training progress")
        print("   • genome_*_analysis.png - Various visualization plots")
        print("   • interactive_genome_browser.html - Interactive visualization")
        
        if alignment_results:
            print("   • sequence_alignment_detailed.png - Detailed alignment visualization")
            print("   • phylogenetic_tree.png - Phylogenetic tree")
        
        print("\n🔬 Advanced Features Implemented:")
        print("   ✅ Multi-model ensemble (LSTM, Transformer, Autoencoder)")
        print("   ✅ Phylogenetic-aware reconstruction")
        print("   ✅ Mutation effect prediction")
        print("   ✅ Confidence scoring system")
        print("   ✅ Progressive sequence alignment")
        print("   ✅ Evolutionary distance calculations")
        print("   ✅ Interactive visualizations")
        print("   ✅ Comprehensive quality analysis")
        
        return output_dir, results, alignment_results, summary_stats
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    print("Genome Sequencing and Reconstruction System")
    print("Advanced AI-Powered Ancient DNA Analysis Pipeline")
    print("=" * 60)
    
    output_dir, results, alignment_results, summary_stats = main()
    
    if output_dir:
        print(f"\n🎯 Analysis completed successfully!")
        print(f"📂 Check the '{output_dir}' directory for all results and visualizations.")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
        sys.exit(1)
