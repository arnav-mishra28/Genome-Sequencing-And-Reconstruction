"""
live_simulation.py
==================
Step-by-step ancient DNA damage simulator that yields one damage event at a time.
Designed for real-time visualization — each call to step() applies a single 
biologically-motivated mutation and returns a structured event dict.

Damage types (applied in biologically-realistic order):
  1. C→T deamination  (position-dependent rate, elevated at fragment ends)
  2. G→A deamination  (reverse-strand complement)
  3. G→T oxidative    (8-oxoguanine)
  4. Random substitution
  5. Small deletion
  6. Small insertion
  7. Large gap (missing segment → N)
"""

import os
import sys
import time
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Generator

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

# ── Constants ─────────────────────────────────────────────────────────────────
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

DAMAGE_COLORS = {
    "deamination_C_to_T": "#FF4444",
    "deamination_G_to_A": "#FF6644",
    "oxidative_G_to_T":   "#FFAA00",
    "random_substitution": "#AA44FF",
    "deletion":            "#FF00AA",
    "insertion":           "#00AAFF",
    "missing_segment":     "#888888",
}

DAMAGE_DESCRIPTIONS = {
    "deamination_C_to_T": "Hydrolytic deamination: C→T (hallmark of ancient DNA)",
    "deamination_G_to_A": "Reverse-strand deamination: G→A",
    "oxidative_G_to_T":   "Oxidative damage: G→T (8-oxoguanine)",
    "random_substitution": "Random substitution from background radiation",
    "deletion":            "Small deletion from strand break",
    "insertion":           "Small insertion from replication slippage",
    "missing_segment":     "Large gap — DNA backbone hydrolysis over millennia",
}


class LiveDamageSimulator:
    """
    Generator-based ancient DNA damage simulator.
    
    Usage:
        sim = LiveDamageSimulator("ACGTACGT...", name="neanderthal")
        
        # Auto mode:
        for event in sim.auto_run(speed=10):
            update_visualization(event)
        
        # Manual mode:
        event = sim.manual_step()
    """

    def __init__(
        self,
        sequence:         str,
        name:             str           = "specimen",
        seed:             int           = 42,
        deamination_rate: float         = 0.08,
        mutation_rate:    float         = 0.02,
        oxidation_rate:   float         = 0.01,
        deletion_rate:    float         = 0.005,
        insertion_rate:   float         = 0.003,
        gap_count:        int           = 5,
        gap_size_range:   Tuple[int,int]= (50, 300),
        strand_bias:      float         = 0.7,
        max_events:       int           = 500,
    ):
        self.name             = name
        self.original_seq     = sequence.upper()
        self.current_seq      = list(self.original_seq)
        self.length           = len(self.current_seq)
        self.seed             = seed
        self.rng              = random.Random(seed)
        self.np_rng           = np.random.default_rng(seed)
        
        # Rates
        self.deamination_rate = deamination_rate
        self.mutation_rate    = mutation_rate
        self.oxidation_rate   = oxidation_rate
        self.deletion_rate    = deletion_rate
        self.insertion_rate   = insertion_rate
        self.gap_count        = gap_count
        self.gap_size_range   = gap_size_range
        self.strand_bias      = strand_bias
        self.max_events       = max_events
        
        # State
        self.events:     List[Dict] = []
        self.step_count: int        = 0
        self.paused:     bool       = False
        self.finished:   bool       = False
        self.speed:      float      = 5.0     # events per second
        
        # Pre-compute the damage plan (queue of planned events)
        self._damage_queue: List[Dict] = []
        self._build_damage_queue()
        
        # Callbacks
        self._on_damage:  Optional[Callable] = None
        self._on_finish:  Optional[Callable] = None
    
    # ── Callbacks ─────────────────────────────────────────────────────────────
    def on_damage(self, callback: Callable):
        """Register a callback: fn(event_dict) called after each damage step."""
        self._on_damage = callback
    
    def on_finish(self, callback: Callable):
        """Register a callback: fn(simulator) called when all damage is done."""
        self._on_finish = callback
    
    # ── Damage rate curve ─────────────────────────────────────────────────────
    @staticmethod
    def _deamination_rate_curve(pos: int, length: int, base_rate: float) -> float:
        dist = min(pos, length - 1 - pos)
        if dist < 5:
            return base_rate * 4.0
        elif dist < 15:
            return base_rate * 2.5 * np.exp(-0.1 * (dist - 5))
        elif dist < 30:
            return base_rate * 1.2
        return base_rate * 0.3
    
    # ── Build the damage plan ─────────────────────────────────────────────────
    def _build_damage_queue(self):
        """Pre-compute all damage events in biologically-ordered sequence."""
        queue = []
        seq = list(self.original_seq)
        length = len(seq)
        
        # Phase 1: Deamination (C→T, G→A) — position dependent
        for i in range(length):
            rate = self._deamination_rate_curve(i, length, self.deamination_rate)
            if seq[i] == "C" and self.rng.random() < rate * self.strand_bias:
                queue.append({
                    "type":     "deamination_C_to_T",
                    "position": i,
                    "original": "C",
                    "mutated":  "T",
                    "phase":    "deamination",
                })
                seq[i] = "T"
            elif seq[i] == "G" and self.rng.random() < rate * (1 - self.strand_bias + 0.3):
                queue.append({
                    "type":     "deamination_G_to_A",
                    "position": i,
                    "original": "G",
                    "mutated":  "A",
                    "phase":    "deamination",
                })
                seq[i] = "A"
        
        # Phase 2: Oxidative damage
        for i in range(length):
            if seq[i] == "G" and self.rng.random() < self.oxidation_rate:
                queue.append({
                    "type":     "oxidative_G_to_T",
                    "position": i,
                    "original": "G",
                    "mutated":  "T",
                    "phase":    "oxidation",
                })
                seq[i] = "T"
        
        # Phase 3: Random substitutions
        for i in range(length):
            if self.rng.random() < self.mutation_rate:
                orig = seq[i]
                choices = [b for b in "ACGT" if b != orig]
                mut = self.rng.choice(choices)
                queue.append({
                    "type":     "random_substitution",
                    "position": i,
                    "original": orig,
                    "mutated":  mut,
                    "phase":    "background_radiation",
                })
                seq[i] = mut
        
        # Phase 4: Small deletions
        for i in range(length):
            if self.rng.random() < self.deletion_rate:
                queue.append({
                    "type":     "deletion",
                    "position": i,
                    "original": seq[i],
                    "mutated":  "",
                    "phase":    "degradation",
                })
        
        # Phase 5: Small insertions
        for i in range(length):
            if self.rng.random() < self.insertion_rate:
                ins = self.rng.choice("ACGT")
                queue.append({
                    "type":     "insertion",
                    "position": i,
                    "original": "",
                    "mutated":  ins,
                    "phase":    "degradation",
                })
        
        # Phase 6: Large gaps
        for g in range(self.gap_count):
            start = self.rng.randint(0, length - 1)
            size  = self.rng.randint(*self.gap_size_range)
            end   = min(start + size, length)
            queue.append({
                "type":     "missing_segment",
                "position": start,
                "end":      end,
                "original": "".join(seq[start:end])[:20] + "…",
                "mutated":  "N" * (end - start),
                "phase":    "physical_degradation",
                "gap_size": end - start,
            })
        
        # Shuffle within phases to add some randomness, but keep phase order
        # Actually — interleave for visual interest, but weight early events
        # to deamination (the most common ancient damage).
        # Limit to max_events
        self._damage_queue = queue[:self.max_events]
    
    # ── Statistics ────────────────────────────────────────────────────────────
    @property
    def stats(self) -> Dict:
        """Return current simulation statistics."""
        seq_str = "".join(self.current_seq)
        n_gaps = seq_str.count("N")
        original_len = len(self.original_seq)
        
        # Count by type
        type_counts = {}
        for e in self.events:
            t = e["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        
        # Compute identity vs original
        matches = sum(1 for a, b in zip(self.original_seq, seq_str)
                      if a == b and a != "N")
        total_valid = sum(1 for c in self.original_seq if c != "N")
        identity = matches / max(total_valid, 1)
        
        return {
            "name":             self.name,
            "step":             self.step_count,
            "total_planned":    len(self._damage_queue),
            "total_applied":    len(self.events),
            "remaining":        len(self._damage_queue) - self.step_count,
            "gaps":             n_gaps,
            "identity":         identity,
            "seq_length":       len(self.current_seq),
            "original_length":  original_len,
            "type_counts":      type_counts,
            "finished":         self.finished,
            "paused":           self.paused,
            "speed":            self.speed,
        }
    
    # ── Core step ─────────────────────────────────────────────────────────────
    def manual_step(self) -> Optional[Dict]:
        """Apply one damage event and return it. Returns None if finished."""
        if self.step_count >= len(self._damage_queue):
            self.finished = True
            if self._on_finish:
                self._on_finish(self)
            return None
        
        event = self._damage_queue[self.step_count].copy()
        event["step"]      = self.step_count
        event["timestamp"] = time.time()
        
        # Apply the damage to current_seq
        pos = event["position"]
        etype = event["type"]
        
        if etype == "missing_segment":
            end = event.get("end", pos + 50)
            end = min(end, len(self.current_seq))
            for j in range(pos, end):
                if j < len(self.current_seq):
                    self.current_seq[j] = "N"
            event["applied"] = True
        elif etype == "deletion":
            if pos < len(self.current_seq):
                # Mark as N instead of removing (keeps coords stable for viz)
                event["original"] = self.current_seq[pos]
                self.current_seq[pos] = "N"
                event["applied"] = True
            else:
                event["applied"] = False
        elif etype == "insertion":
            # For visualization stability, replace nearest non-N with new base
            if pos < len(self.current_seq):
                event["applied"] = True
                # Don't actually insert (keeps array length stable for 3D coords)
            else:
                event["applied"] = False
        else:
            # Substitution types
            if pos < len(self.current_seq):
                event["original"] = self.current_seq[pos]
                self.current_seq[pos] = event["mutated"]
                event["applied"] = True
            else:
                event["applied"] = False
        
        # Add color info
        event["color"] = DAMAGE_COLORS.get(etype, "#FFFFFF")
        event["description"] = DAMAGE_DESCRIPTIONS.get(etype, "Unknown damage")
        
        self.events.append(event)
        self.step_count += 1
        
        if self._on_damage:
            self._on_damage(event)
        
        return event
    
    # ── Auto-run generator ────────────────────────────────────────────────────
    def auto_run(self, speed: float = 5.0) -> Generator[Dict, None, None]:
        """
        Generator that yields damage events at `speed` events/second.
        Respects self.paused flag.
        """
        self.speed = speed
        while not self.finished:
            if self.paused:
                time.sleep(0.05)
                continue
            
            event = self.manual_step()
            if event is None:
                break
            yield event
            
            if self.speed > 0:
                time.sleep(1.0 / self.speed)
    
    # ── Control ───────────────────────────────────────────────────────────────
    def toggle_pause(self):
        self.paused = not self.paused
    
    def set_speed(self, speed: float):
        self.speed = max(0.5, min(speed, 200.0))
    
    def speed_up(self):
        self.set_speed(self.speed * 1.5)
    
    def slow_down(self):
        self.set_speed(self.speed / 1.5)
    
    def reset(self):
        """Reset simulation to initial state."""
        self.current_seq = list(self.original_seq)
        self.events.clear()
        self.step_count = 0
        self.finished = False
        self.paused = False
        self._build_damage_queue()
    
    def get_current_sequence(self) -> str:
        return "".join(self.current_seq)
    
    def get_damage_positions(self) -> List[int]:
        """Return all positions that have been damaged."""
        return [e["position"] for e in self.events if e.get("applied", False)]
    
    def get_recent_events(self, n: int = 5) -> List[Dict]:
        """Return the last n events."""
        return self.events[-n:]


# ── Convenience: create simulator from species name ───────────────────────────
def create_simulator_from_species(
    species_name: str,
    max_bases:    int   = 2000,
    max_events:   int   = 500,
    seed:         int   = 42,
) -> LiveDamageSimulator:
    """
    Load a species sequence from disk and create a LiveDamageSimulator.
    Falls back to a synthetic sequence if the real data isn't available.
    """
    from config.settings import SEQ_DIR
    
    meta_path = os.path.join(SEQ_DIR, "metadata.json")
    sequence = None
    
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            metadata = json.load(f)
        if species_name in metadata:
            fasta_path = metadata[species_name].get("path", "")
            if os.path.exists(fasta_path):
                from data.fetch_sequences import load_fasta
                records = load_fasta(fasta_path)
                sequence = next(iter(records.values()))[:max_bases]
    
    if sequence is None:
        # Generate realistic synthetic mtDNA-like sequence
        rng = random.Random(seed)
        # mtDNA has ~44% GC content
        weights = [0.28, 0.22, 0.22, 0.28]  # A, C, G, T
        bases = random.choices("ACGT", weights=weights, k=max_bases)
        sequence = "".join(bases)
        print(f"  ⚠ Using synthetic sequence for '{species_name}' ({max_bases} bp)")
    
    return LiveDamageSimulator(
        sequence=sequence,
        name=species_name,
        seed=seed,
        max_events=max_events,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧬 Live Damage Simulator — Test Run")
    print("=" * 50)
    
    seq = "ACGTACGTACGT" * 50  # 600 bp test sequence
    sim = LiveDamageSimulator(seq, name="test_specimen", max_events=30)
    
    print(f"  Original: {seq[:60]}...")
    print(f"  Planned events: {len(sim._damage_queue)}")
    print()
    
    for event in sim.auto_run(speed=100):
        stats = sim.stats
        print(f"  Step {event['step']:3d} │ {event['type']:25s} │ "
              f"pos={event['position']:4d} │ "
              f"{event.get('original','?')}→{event.get('mutated','?')} │ "
              f"identity={stats['identity']:.3f}")
    
    print(f"\n  Final: {sim.get_current_sequence()[:60]}...")
    print(f"  Stats: {sim.stats}")
