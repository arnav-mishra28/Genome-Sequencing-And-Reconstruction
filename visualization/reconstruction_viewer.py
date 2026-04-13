"""
reconstruction_viewer.py
========================
Real-time 3D DNA Reconstruction Viewer — PyOpenGL + Pygame.

A GPU-accelerated interactive viewer that shows the AI reconstructing
damaged DNA in real-time. Features:

  🎨 Visual:
    - Rotating 3D double helix with per-base nucleotide coloring
    - Damaged bases pulse red/gray; gaps show as broken rungs
    - Reconstruction animation: gray→colored glow burst as AI fills bases
    - Confidence heatmap overlay: bright=high, dim=low confidence
    - Smooth backbone ribbons with hydrogen bond rungs
    - Particle burst effects on reconstruction events

  📊 HUD:
    - Reconstruction progress bar
    - Live confidence chart
    - Per-base stats overlay
    - Phase indicator (AE / Fusion / BERT)
    - Species name and metrics

  🎮 Controls:
    - Mouse drag: rotate view
    - Scroll: zoom in/out
    - Space: pause/resume reconstruction
    - +/-: adjust speed
    - Tab: cycle species
    - C: toggle confidence overlay
    - S: save snapshot
    - Q: quit
"""

import os
import sys
import math
import time
import random
import numpy as np
from typing import List, Dict, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.dirname(_THIS_DIR)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

# ── Check dependencies ───────────────────────────────────────────────────────
_HAS_OPENGL = False
_HAS_PYGAME = False

try:
    import pygame
    from pygame.locals import *
    _HAS_PYGAME = True
except ImportError:
    pass

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _HAS_OPENGL = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Color Palette
# ═══════════════════════════════════════════════════════════════════════════════
BASE_COLORS = {
    "A": (0.0, 0.8, 0.27, 1.0),    # green
    "C": (0.0, 0.4, 1.0, 1.0),     # blue
    "G": (1.0, 0.53, 0.0, 1.0),    # orange
    "T": (1.0, 0.13, 0.13, 1.0),   # red
    "N": (0.33, 0.33, 0.33, 0.5),  # dim gray
}

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

CONF_COLORS = {   # confidence level colors (for HUD)
    "HIGH":   (0.0, 1.0, 0.53, 1.0),
    "MEDIUM": (1.0, 0.8, 0.0, 1.0),
    "LOW":    (1.0, 0.2, 0.2, 1.0),
}

DARK_BG = (0.04, 0.04, 0.06)
ACCENT  = (0.345, 0.651, 1.0)
GLOW    = (0.5, 0.9, 1.0)

PHASE_COLORS = {
    "denoising":              (0.0, 0.8, 1.0),
    "fusion_reconstruction":  (0.6, 0.2, 1.0),
    "bert_fill":              (1.0, 0.6, 0.0),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Particle Effect
# ═══════════════════════════════════════════════════════════════════════════════
class Particle:
    """A single particle for burst effects."""
    def __init__(self, x, y, z, color, lifetime=1.0):
        self.x, self.y, self.z = x, y, z
        self.color = color
        self.lifetime = lifetime
        self.age = 0.0
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.01, 0.05)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.vz = random.uniform(-0.02, 0.02)
        self.size = random.uniform(2, 5)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.vx *= 0.95
        self.vy *= 0.95
        self.vz *= 0.95

    @property
    def alive(self):
        return self.age < self.lifetime

    @property
    def alpha(self):
        return max(0, 1.0 - self.age / self.lifetime)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helix Renderer (OpenGL)
# ═══════════════════════════════════════════════════════════════════════════════
class HelixRenderer:
    """Renders 3D DNA double helix with OpenGL."""

    def __init__(self, max_bases=300):
        self.max_bases = max_bases
        self.sequence = []
        self.confidences = []
        self.damaged_set = set()
        self.reconstructed_set = set()
        self.flash_positions = {}  # pos → remaining flash frames
        self.particles = []
        self.show_confidence = True

        # Helix geometry
        self.turns_per_base = 0.1
        self.rise_per_base  = 0.34
        self.radius         = 1.0

    def set_sequence(self, seq: str, confidences: List[float] = None):
        self.sequence = list(seq.upper()[:self.max_bases])
        n = len(self.sequence)
        if confidences:
            self.confidences = confidences[:self.max_bases]
        else:
            self.confidences = [0.5] * n
        self.damaged_set = {i for i, b in enumerate(self.sequence) if b == "N"}

    def apply_reconstruction(self, pos: int, new_base: str, confidence: float):
        if pos < 0 or pos >= len(self.sequence):
            return
        self.sequence[pos] = new_base.upper()
        self.confidences[pos] = confidence
        self.reconstructed_set.add(pos)
        self.damaged_set.discard(pos)
        self.flash_positions[pos] = 30  # flash for 30 frames

        # Spawn particles
        t = 2 * math.pi * self.turns_per_base * pos
        z = self.rise_per_base * pos
        x = self.radius * math.cos(t)
        y = self.radius * math.sin(t)
        color = BASE_COLORS.get(new_base.upper(), BASE_COLORS["N"])
        for _ in range(8):
            self.particles.append(Particle(x, y, z, color, lifetime=0.8))

    def render(self, rotation_angle: float, zoom: float,
               elevation: float = 20.0):
        """Render the full helix scene."""
        n = len(self.sequence)
        if n == 0:
            return

        glPushMatrix()
        glTranslatef(0, 0, -zoom)
        glRotatef(elevation, 1, 0, 0)
        glRotatef(rotation_angle, 0, 0, 1)
        # Center vertically
        total_height = self.rise_per_base * n
        glTranslatef(0, 0, -total_height / 2)

        t_arr = np.linspace(0, 2 * math.pi * self.turns_per_base * n, n)
        z_arr = np.linspace(0, self.rise_per_base * n, n)
        x1 = self.radius * np.cos(t_arr)
        y1 = self.radius * np.sin(t_arr)
        x2 = -x1
        y2 = -y1

        # ── Backbone ribbons ─────────────────────────────────────────────────
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for i in range(n):
            c = self._get_color(i)
            glColor4f(c[0] * 0.5, c[1] * 0.5, c[2] * 0.5, 0.6)
            glVertex3f(x1[i], y1[i], z_arr[i])
        glEnd()

        glBegin(GL_LINE_STRIP)
        for i in range(n):
            comp = COMPLEMENT.get(self.sequence[i], "N")
            c = BASE_COLORS.get(comp, BASE_COLORS["N"])
            glColor4f(c[0] * 0.4, c[1] * 0.4, c[2] * 0.4, 0.4)
            glVertex3f(x2[i], y2[i], z_arr[i])
        glEnd()

        # ── Hydrogen bonds (every 2nd base) ──────────────────────────────────
        glLineWidth(1.0)
        for i in range(0, n, 2):
            c = self._get_color(i)
            alpha = 0.15 if self.sequence[i] == "N" else 0.3
            glColor4f(c[0], c[1], c[2], alpha)
            glBegin(GL_LINES)
            glVertex3f(x1[i], y1[i], z_arr[i])
            glVertex3f(x2[i], y2[i], z_arr[i])
            glEnd()

        # ── Nucleotide spheres (strand 1) ────────────────────────────────────
        for i in range(n):
            c = self._get_color(i)
            size = self._get_size(i)
            glPushMatrix()
            glTranslatef(x1[i], y1[i], z_arr[i])

            # Glow effect for recently reconstructed
            if i in self.flash_positions:
                glow_alpha = self.flash_positions[i] / 30.0
                glColor4f(
                    min(1, c[0] + 0.5 * glow_alpha),
                    min(1, c[1] + 0.5 * glow_alpha),
                    min(1, c[2] + 0.5 * glow_alpha),
                    1.0
                )
                size *= (1.0 + 0.5 * glow_alpha)
            elif self.show_confidence and i < len(self.confidences):
                conf = self.confidences[i]
                glColor4f(c[0] * conf, c[1] * conf, c[2] * conf,
                          0.3 + 0.7 * conf)
            else:
                glColor4f(*c)

            self._draw_sphere(size)
            glPopMatrix()

        # ── Nucleotide spheres (strand 2 — complement) ───────────────────────
        for i in range(n):
            comp = COMPLEMENT.get(self.sequence[i], "N")
            c = BASE_COLORS.get(comp, BASE_COLORS["N"])
            glPushMatrix()
            glTranslatef(x2[i], y2[i], z_arr[i])
            glColor4f(c[0] * 0.6, c[1] * 0.6, c[2] * 0.6, 0.5)
            self._draw_sphere(0.03)
            glPopMatrix()

        # ── Particles ────────────────────────────────────────────────────────
        glPointSize(3.0)
        glBegin(GL_POINTS)
        for p in self.particles:
            glColor4f(p.color[0], p.color[1], p.color[2], p.alpha)
            glVertex3f(p.x, p.y, p.z)
        glEnd()

        glPopMatrix()

    def update(self, dt: float):
        """Update animations."""
        # Decay flash
        dead_flashes = []
        for pos in list(self.flash_positions):
            self.flash_positions[pos] -= 1
            if self.flash_positions[pos] <= 0:
                dead_flashes.append(pos)
        for pos in dead_flashes:
            del self.flash_positions[pos]

        # Update particles
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive]

    def _get_color(self, idx):
        base = self.sequence[idx] if idx < len(self.sequence) else "N"
        return BASE_COLORS.get(base, BASE_COLORS["N"])

    def _get_size(self, idx):
        base = self.sequence[idx] if idx < len(self.sequence) else "N"
        if base == "N":
            return 0.02
        if idx in self.flash_positions:
            return 0.06
        return 0.04

    @staticmethod
    def _draw_sphere(radius, slices=8, stacks=6):
        """Draw a small sphere using quadrics."""
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluSphere(quad, radius, slices, stacks)
        gluDeleteQuadric(quad)


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD Renderer (2D overlay)
# ═══════════════════════════════════════════════════════════════════════════════
class HUDRenderer:
    """2D overlay for stats, progress, and confidence bars."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def render(self, stats: Dict, engine, phase_color=(0.5, 0.9, 1.0)):
        """Draw HUD elements."""
        self._setup_2d()

        # ── Title ────────────────────────────────────────────────────────────
        self._draw_text(
            f"🧬 LIVE DNA RECONSTRUCTION — {stats.get('species', '')}",
            20, self.height - 30,
            color=ACCENT, size=18,
        )

        # ── Progress bar ─────────────────────────────────────────────────────
        progress = stats.get("progress", 0)
        bar_w = 300
        bar_h = 20
        bar_x = 20
        bar_y = self.height - 70

        # Background
        self._draw_rect(bar_x, bar_y, bar_w, bar_h, (0.15, 0.15, 0.2, 0.8))
        # Fill
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pc = phase_color
            self._draw_rect(bar_x, bar_y, fill_w, bar_h,
                           (pc[0], pc[1], pc[2], 0.9))
        # Border
        self._draw_rect_outline(bar_x, bar_y, bar_w, bar_h,
                                (0.5, 0.5, 0.6, 0.8))
        # Label
        self._draw_text(
            f"{progress:.1%} ({stats.get('gaps_filled', 0)}/{stats.get('total_gaps', 0)} gaps)",
            bar_x + bar_w + 10, bar_y + 3,
            color=(0.8, 0.8, 0.8),
        )

        # ── Stats panel ──────────────────────────────────────────────────────
        panel_y = self.height - 110
        self._draw_rect(20, panel_y - 80, 280, 75, (0.08, 0.08, 0.12, 0.85))
        self._draw_rect_outline(20, panel_y - 80, 280, 75,
                                (0.3, 0.3, 0.4, 0.5))

        lines = [
            f"Bases: {stats.get('total_bases', 0):,}",
            f"Gaps remaining: {stats.get('gaps_remaining', 0):,}",
            f"Mean confidence: {stats.get('mean_confidence', 0):.3f}",
            f"Steps: {stats.get('steps', 0):,}",
        ]
        for i, line in enumerate(lines):
            self._draw_text(line, 30, panel_y - 20 - i * 17,
                           color=(0.7, 0.7, 0.8))

        # ── Phase indicator ──────────────────────────────────────────────────
        phase = "IDLE"
        if engine and hasattr(engine, 'events') and engine.events:
            phase = engine.events[-1].phase.upper()
        self._draw_text(
            f"▶ {phase}",
            20, self.height - 200,
            color=phase_color, size=14,
        )

        # ── Confidence mini-chart ────────────────────────────────────────────
        if engine and hasattr(engine, 'confidences'):
            self._draw_confidence_bars(engine.confidences, engine)

        # ── Controls ─────────────────────────────────────────────────────────
        controls = "[SPACE] Pause  [+/-] Speed  [C] Confidence  [S] Save  [Q] Quit"
        self._draw_text(controls, 20, 15, color=(0.4, 0.4, 0.5), size=10)

        self._teardown_2d()

    def _draw_confidence_bars(self, confidences, engine):
        """Draw a mini confidence histogram in bottom-right."""
        if not confidences:
            return

        chart_w = 250
        chart_h = 100
        chart_x = self.width - chart_w - 20
        chart_y = 40

        # Background
        self._draw_rect(chart_x, chart_y, chart_w, chart_h,
                        (0.08, 0.08, 0.12, 0.85))
        self._draw_rect_outline(chart_x, chart_y, chart_w, chart_h,
                                (0.3, 0.3, 0.4, 0.5))

        # Title
        self._draw_text("Per-Base Confidence", chart_x + 5, chart_y + chart_h - 15,
                        color=(0.6, 0.6, 0.7), size=10)

        # Downsample confidences to fit chart width
        n_bars = min(len(confidences), chart_w - 10)
        step = max(1, len(confidences) // n_bars)
        bar_w = max(1, (chart_w - 10) / n_bars)

        for i in range(n_bars):
            idx = i * step
            if idx >= len(confidences):
                break
            c = confidences[idx]
            bar_h_px = int(c * (chart_h - 25))

            if c >= 0.85:
                color = (0.0, 1.0, 0.53, 0.8)
            elif c >= 0.5:
                color = (1.0, 0.8, 0.0, 0.8)
            else:
                color = (1.0, 0.2, 0.2, 0.8)

            bx = chart_x + 5 + i * bar_w
            by = chart_y + 5
            self._draw_rect(bx, by, max(1, bar_w - 1), bar_h_px, color)

    def _setup_2d(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

    def _teardown_2d(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_rect(self, x, y, w, h, color):
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def _draw_rect_outline(self, x, y, w, h, color):
        glColor4f(*color)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def _draw_text(self, text, x, y, color=(1, 1, 1), size=12):
        """Draw text using pygame surface → OpenGL texture."""
        # Use pygame font rendering
        if not _HAS_PYGAME:
            return
        try:
            font = pygame.font.SysFont("consolas", size)
        except:
            font = pygame.font.SysFont(None, size)

        c = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        surface = font.render(str(text), True, c)
        data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_size()

        glRasterPos2f(x, y)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
class ReconstructionViewer:
    """
    Main real-time 3D DNA reconstruction viewer.

    Combines:
      - HelixRenderer: 3D helix
      - HUDRenderer:   2D overlay
      - ReconstructionEngine: AI model driver
    """

    def __init__(
        self,
        engine,
        width:     int   = 1600,
        height:    int   = 900,
        max_bases: int   = 300,
        title:     str   = "🧬 Live DNA Reconstruction",
    ):
        self.engine    = engine
        self.width     = width
        self.height    = height
        self.title     = title

        self.helix     = HelixRenderer(max_bases)
        self.hud       = HUDRenderer(width, height)

        # Camera
        self.rotation  = 0.0
        self.elevation = 20.0
        self.zoom      = 8.0
        self.auto_rotate = True
        self.rotation_speed = 0.3

        # Mouse
        self.mouse_down   = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Reconstruction
        self.recon_gen     = None
        self.paused        = False
        self.speed         = 10.0   # events per second
        self.show_conf     = True
        self.last_recon_time = 0

        # Phase color
        self.phase_color = (0.5, 0.9, 1.0)

    def launch(self):
        """Start the viewer (blocking)."""
        if not _HAS_PYGAME or not _HAS_OPENGL:
            print("=" * 65)
            print("  ⚠ PyOpenGL or Pygame not installed.")
            print("  Falling back to matplotlib-based viewer.")
            print("=" * 65)
            self._fallback_matplotlib()
            return

        pygame.init()
        pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL | RESIZABLE,
        )
        pygame.display.set_caption(self.title)

        self._init_gl()
        self._init_helix()
        self.recon_gen = self.engine.reconstruct()

        clock = pygame.time.Clock()
        running = True

        print("  🎮 Viewer launched — use controls to interact")
        print("  [SPACE] Pause  [+/-] Speed  [C] Confidence  [Q] Quit")

        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_down = True
                        self.last_mouse_x, self.last_mouse_y = event.pos
                    elif event.button == 4:
                        self.zoom = max(3, self.zoom - 0.5)
                    elif event.button == 5:
                        self.zoom = min(30, self.zoom + 0.5)
                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down = False
                elif event.type == MOUSEMOTION:
                    if self.mouse_down:
                        dx = event.pos[0] - self.last_mouse_x
                        dy = event.pos[1] - self.last_mouse_y
                        self.rotation += dx * 0.3
                        self.elevation = max(-90, min(90,
                            self.elevation + dy * 0.3))
                        self.last_mouse_x, self.last_mouse_y = event.pos
                        self.auto_rotate = False
                elif event.type == VIDEORESIZE:
                    self.width, self.height = event.size
                    pygame.display.set_mode(
                        (self.width, self.height),
                        DOUBLEBUF | OPENGL | RESIZABLE,
                    )
                    glViewport(0, 0, self.width, self.height)
                    self.hud = HUDRenderer(self.width, self.height)
                    self._setup_projection()

            # ── Update ────────────────────────────────────────────────────────
            self._update(dt)

            # ── Render ────────────────────────────────────────────────────────
            self._render()

            pygame.display.flip()

        pygame.quit()

    def _init_gl(self):
        """Initialize OpenGL state."""
        glClearColor(*DARK_BG, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        self._setup_projection()

    def _setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / max(1, self.height)
        gluPerspective(45, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def _init_helix(self):
        """Set up the helix with the damaged sequence."""
        seq = self.engine.get_current_sequence()
        confs = self.engine.get_confidence_array()
        self.helix.set_sequence(seq, confs)
        self.helix.show_confidence = self.show_conf

    def _handle_key(self, key) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == K_q or key == K_ESCAPE:
            return False
        elif key == K_SPACE:
            self.paused = not self.paused
        elif key in (K_PLUS, K_EQUALS):
            self.speed = min(200, self.speed * 1.5)
        elif key == K_MINUS:
            self.speed = max(0.5, self.speed / 1.5)
        elif key == K_c:
            self.show_conf = not self.show_conf
            self.helix.show_confidence = self.show_conf
        elif key == K_r:
            self.auto_rotate = not self.auto_rotate
        elif key == K_s:
            self._save_snapshot()
        return True

    def _update(self, dt: float):
        """Update simulation and animations."""
        # Auto-rotation
        if self.auto_rotate:
            self.rotation += self.rotation_speed

        # Helix animations
        self.helix.update(dt)

        # Run reconstruction steps
        if not self.paused and self.recon_gen is not None:
            now = time.time()
            interval = 1.0 / max(0.1, self.speed)
            if now - self.last_recon_time >= interval:
                try:
                    event = next(self.recon_gen)
                    self.helix.apply_reconstruction(
                        event.position,
                        event.predicted_base,
                        event.confidence,
                    )
                    # Update confidences
                    if event.position < len(self.helix.confidences):
                        self.helix.confidences[event.position] = event.confidence
                    # Phase color
                    self.phase_color = PHASE_COLORS.get(
                        event.phase, (0.5, 0.9, 1.0)
                    )
                    self.last_recon_time = now
                except StopIteration:
                    self.recon_gen = None

    def _render(self):
        """Render frame."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 3D helix
        self.helix.render(self.rotation, self.zoom, self.elevation)

        # 2D HUD
        stats = self.engine.stats if self.engine else {}
        if self.paused:
            stats["_paused"] = True
        self.hud.render(stats, self.engine, self.phase_color)

    def _save_snapshot(self):
        """Save current frame as PNG."""
        try:
            from config.settings import VIZ_DIR
            path = os.path.join(
                VIZ_DIR,
                f"reconstruction_snapshot_{int(time.time())}.png"
            )

            data = glReadPixels(0, 0, self.width, self.height,
                                GL_RGBA, GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(
                data, (self.width, self.height), "RGBA", True
            )
            pygame.image.save(surface, path)
            print(f"  📸 Snapshot saved: {path}")
        except Exception as e:
            print(f"  ⚠ Snapshot failed: {e}")

    # ── Matplotlib fallback ───────────────────────────────────────────────────
    def _fallback_matplotlib(self):
        """
        Fallback to matplotlib-based viewer when OpenGL is not available.
        Uses the existing LiveViewer infrastructure.
        """
        print("  Using matplotlib 3D viewer (install PyOpenGL + pygame for better visuals)")

        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(18, 10), facecolor="#0a0a0a",
                         num="🧬 Live DNA Reconstruction")

        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1],
                              wspace=0.08,
                              left=0.03, right=0.97, top=0.92, bottom=0.08)

        # 3D helix axis
        ax3d = fig.add_subplot(gs[0], projection="3d", facecolor="#0a0a0a")
        ax3d.set_title("3D DNA — Live Reconstruction",
                       fontsize=12, color="#58a6ff", pad=8)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor("#222")
        ax3d.yaxis.pane.set_edgecolor("#222")
        ax3d.zaxis.pane.set_edgecolor("#222")
        ax3d.tick_params(colors="#666", labelsize=5)

        # Confidence axis
        ax_conf = fig.add_subplot(gs[1], facecolor="#0d1117")
        ax_conf.set_title("Per-Base Confidence",
                          fontsize=11, color="#58a6ff")

        fig.suptitle(
            f"🧬  LIVE DNA RECONSTRUCTION  —  {self.engine.species_name}",
            fontsize=16, color="white", fontweight="bold", y=0.98,
        )

        status_text = fig.text(
            0.02, 0.04, "Initializing...",
            fontsize=9, color="#58a6ff", fontfamily="monospace",
        )

        controls_text = fig.text(
            0.5, 0.02,
            "[SPACE] Pause  [+] Speed Up  [-] Slow Down  [Q] Quit",
            ha="center", fontsize=8, color="#666",
            fontfamily="monospace",
        )

        MPL_BASE_COLORS = {
            "A": "#00CC44", "C": "#0066FF",
            "G": "#FF8800", "T": "#FF2222", "N": "#555555",
        }

        self.recon_gen = self.engine.reconstruct()
        azimuth = [45.0]
        recon_active = [True]

        def update(frame):
            # Run reconstruction step
            if recon_active[0] and not self.paused:
                try:
                    for _ in range(max(1, int(self.speed / 20))):
                        event = next(self.recon_gen)
                except StopIteration:
                    recon_active[0] = False

            # Get current state
            seq = self.engine.current_seq
            confs = self.engine.confidences
            n = min(len(seq), 200)

            # ── Update 3D helix ──────────────────────────────────────────────
            ax3d.clear()
            ax3d.set_facecolor("#0a0a0a")
            ax3d.xaxis.pane.fill = False
            ax3d.yaxis.pane.fill = False
            ax3d.zaxis.pane.fill = False

            t = np.linspace(0, 2 * np.pi * 0.1 * n, n)
            z = np.linspace(0, 0.34 * n, n)
            x1 = np.cos(t)
            y1 = np.sin(t)

            colors = [MPL_BASE_COLORS.get(seq[i], "#555") for i in range(n)]
            sizes = [12 if seq[i] == "N" else 40 for i in range(n)]

            ax3d.scatter(x1, y1, z, c=colors, s=sizes,
                        alpha=0.9, edgecolors="white", linewidths=0.3)
            ax3d.plot(x1, y1, z, color="gray", alpha=0.2, linewidth=0.8)
            ax3d.plot(-x1, -y1, z, color="gray", alpha=0.15, linewidth=0.6)

            azimuth[0] = (azimuth[0] + 0.5) % 360
            ax3d.view_init(elev=20, azim=azimuth[0])
            ax3d.tick_params(colors="#666", labelsize=5)

            # ── Update confidence chart ──────────────────────────────────────
            ax_conf.clear()
            ax_conf.set_facecolor("#0d1117")
            ax_conf.set_title("Per-Base Confidence",
                              fontsize=11, color="#58a6ff")

            if confs:
                x = np.arange(min(len(confs), 500))
                c = np.array(confs[:500])
                bar_colors = []
                for v in c:
                    if v >= 0.85:
                        bar_colors.append("#00ff88")
                    elif v >= 0.5:
                        bar_colors.append("#ffcc00")
                    else:
                        bar_colors.append("#ff3333")
                ax_conf.bar(x, c, color=bar_colors, width=1.0, alpha=0.7)
                ax_conf.set_ylim(0, 1.05)
                ax_conf.set_ylabel("Confidence", color="#888", fontsize=9)
                ax_conf.set_xlabel("Position", color="#888", fontsize=9)
                ax_conf.axhline(y=0.85, color="#00ff88", alpha=0.3,
                               linestyle="--", linewidth=0.8)
                ax_conf.axhline(y=0.5, color="#ff3333", alpha=0.3,
                               linestyle="--", linewidth=0.8)
                ax_conf.tick_params(colors="#666", labelsize=7)
                for spine in ax_conf.spines.values():
                    spine.set_color("#333")

            # ── Update status ────────────────────────────────────────────────
            stats = self.engine.stats
            pause_str = " [PAUSED]" if self.paused else ""
            done_str = " [DONE]" if not recon_active[0] else ""
            status_text.set_text(
                f"Progress: {stats['progress']:.1%} | "
                f"Gaps: {stats['gaps_remaining']:,} remaining | "
                f"Confidence: {stats['mean_confidence']:.3f} | "
                f"Steps: {stats['steps']:,}"
                f"{pause_str}{done_str}"
            )

        def on_key(event):
            if event.key == " ":
                self.paused = not self.paused
            elif event.key in ("+", "="):
                self.speed = min(200, self.speed * 1.5)
            elif event.key == "-":
                self.speed = max(0.5, self.speed / 1.5)
            elif event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        anim = FuncAnimation(fig, update, interval=50, blit=False,
                             cache_frame_data=False)

        try:
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
        except Exception:
            pass

        plt.show()

        # Print final summary
        stats = self.engine.stats
        print("\n" + "=" * 65)
        print("  📊 RECONSTRUCTION SUMMARY")
        print("=" * 65)
        print(f"  Species:        {stats['species']}")
        print(f"  Total bases:    {stats['total_bases']:,}")
        print(f"  Gaps filled:    {stats['gaps_filled']:,} / {stats['total_gaps']:,}")
        print(f"  Gaps remaining: {stats['gaps_remaining']:,}")
        print(f"  Progress:       {stats['progress']:.1%}")
        print(f"  Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"  Total steps:    {stats['steps']:,}")
        print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
#  Launch function
# ═══════════════════════════════════════════════════════════════════════════════
def launch_reconstruction_viewer(
    species_name: str     = "neanderthal_mtDNA",
    damaged_seq:  str     = None,
    max_bases:    int     = 300,
    speed:        float   = 10.0,
    width:        int     = 1600,
    height:       int     = 900,
):
    """
    Main entry point for the real-time 3D reconstruction viewer.

    Args:
        species_name: species to reconstruct
        damaged_seq:  custom damaged sequence (optional)
        max_bases:    max bases for 3D view
        speed:        reconstruction events per second
        width:        window width
        height:       window height
    """
    from visualization.reconstruction_engine import create_reconstruction_engine

    print("\n" + "=" * 65)
    print("  🧬  REAL-TIME 3D DNA RECONSTRUCTION VIEWER")
    print("=" * 65)

    engine = create_reconstruction_engine(
        species_name=species_name,
        damaged_seq=damaged_seq,
        max_bases=max_bases,
    )

    stats = engine.stats
    print(f"  Species:     {species_name}")
    print(f"  Seq length:  {stats['total_bases']:,} bp")
    print(f"  Gaps to fill: {stats['total_gaps']:,}")
    print(f"  Speed:       {speed} events/sec")
    print(f"  Renderer:    {'PyOpenGL + Pygame' if _HAS_OPENGL and _HAS_PYGAME else 'matplotlib (fallback)'}")
    print()
    print("  Controls:")
    print("    [Space]  Pause / Resume")
    print("    [+/=]    Speed up")
    print("    [-]      Slow down")
    print("    [C]      Toggle confidence overlay")
    print("    [Mouse]  Drag to rotate, scroll to zoom")
    print("    [S]      Save snapshot")
    print("    [Q]      Quit")
    print("=" * 65)

    viewer = ReconstructionViewer(
        engine=engine,
        width=width,
        height=height,
        max_bases=max_bases,
    )
    viewer.speed = speed
    viewer.launch()

    return engine


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    launch_reconstruction_viewer(
        species_name="neanderthal_mtDNA",
        speed=15.0,
        max_bases=200,
    )
