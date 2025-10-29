"""
Flappy Bird Game Engine
Core game mechanics and rendering using pygame.
"""

import pygame
import random
import sys
import os

class Bird:
    """Represents the player-controlled bird."""
    
    def __init__(self, x, y, color, jump_strength):
        self.x = x
        self.y = y
        self.velocity = 0
        self.color = color
        self.jump_strength = jump_strength
        self.width = 34
        self.height = 24
        
    def jump(self):
        """Apply upward impulse."""
        self.velocity = self.jump_strength
        
    def update(self, gravity):
        """Update bird position based on gravity and velocity."""
        # Apply gravity (gravity is configured as a per-frame-ish value).
        # Keep velocities as floats for smooth motion and clamp maximum fall speed
        self.velocity += float(gravity)
        # clamp falling speed to avoid tunneling into the ground
        MAX_FALL_SPEED = 20.0
        if self.velocity > MAX_FALL_SPEED:
            self.velocity = MAX_FALL_SPEED
        self.y += self.velocity
        
    def draw(self, screen):
        """Render bird as a simple ellipse."""
        # Main body
        pygame.draw.ellipse(screen, self.color, 
                          (self.x, self.y, self.width, self.height))
        
        # Wing
        wing_color = tuple(max(c - 30, 0) for c in self.color)
        pygame.draw.ellipse(screen, wing_color,
                          (self.x + 5, self.y + 8, 15, 10))
        
        # Eye
        pygame.draw.circle(screen, (255, 255, 255), 
                         (int(self.x + 25), int(self.y + 10)), 5)
        pygame.draw.circle(screen, (0, 0, 0), 
                         (int(self.x + 26), int(self.y + 10)), 3)
        
        # Beak
        beak_points = [
            (self.x + self.width, self.y + 12),
            (self.x + self.width + 8, self.y + 10),
            (self.x + self.width, self.y + 16)
        ]
        pygame.draw.polygon(screen, (255, 140, 0), beak_points)
        
    def get_rect(self):
        """Return collision rectangle."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Pipe:
    """Represents a pair of pipes (top and bottom)."""
    
    def __init__(self, x, gap, color, speed, screen_height):
        self.x = x
        self.gap = gap
        self.color = color
        self.speed = speed
        self.width = 70
        self.screen_height = screen_height
        
        # Randomize gap position
        min_height = 100
        max_height = screen_height - gap - 100
        self.gap_y = random.randint(min_height, max_height)
        self.scored = False
        
    def update(self):
        """Move pipe leftward."""
        self.x -= self.speed
        
    def draw(self, screen):
        """Render top and bottom pipes with 3D effect."""
        # Top pipe
        pygame.draw.rect(screen, self.color, 
                        (self.x, 0, self.width, self.gap_y))
        
        # Top pipe cap
        cap_height = 25
        pygame.draw.rect(screen, self.color,
                        (self.x - 5, self.gap_y - cap_height, self.width + 10, cap_height))
        
        # Bottom pipe
        pygame.draw.rect(screen, self.color,
                        (self.x, self.gap_y + self.gap, 
                         self.width, self.screen_height - self.gap_y - self.gap))
        
        # Bottom pipe cap
        pygame.draw.rect(screen, self.color,
                        (self.x - 5, self.gap_y + self.gap, self.width + 10, cap_height))
        
        # Pipe highlights for 3D effect
        highlight = tuple(min(c + 30, 255) for c in self.color)
        shadow = tuple(max(c - 30, 0) for c in self.color)
        
        # Top pipe highlights
        pygame.draw.rect(screen, highlight, 
                        (self.x, 0, 5, self.gap_y))
        pygame.draw.rect(screen, shadow,
                        (self.x + self.width - 5, 0, 5, self.gap_y))
        
        # Bottom pipe highlights
        pygame.draw.rect(screen, highlight,
                        (self.x, self.gap_y + self.gap, 5, 
                         self.screen_height - self.gap_y - self.gap))
        pygame.draw.rect(screen, shadow,
                        (self.x + self.width - 5, self.gap_y + self.gap, 5,
                         self.screen_height - self.gap_y - self.gap))
        
    def collides_with(self, bird):
        """Check collision with bird."""
        bird_rect = bird.get_rect()
        
        # Top pipe collision
        top_pipe = pygame.Rect(self.x - 5, 0, self.width + 10, self.gap_y)
        
        # Bottom pipe collision
        bottom_pipe = pygame.Rect(self.x - 5, self.gap_y + self.gap, 
                                 self.width + 10, 
                                 self.screen_height - self.gap_y - self.gap)
        
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)
        
    def is_offscreen(self):
        """Check if pipe has moved off screen."""
        return self.x + self.width < 0

class FlappyGame:
    """Main game controller."""

    def __init__(self, config=None, render=True):
        """Initialize game with configuration parameters.

        Args:
            config: configuration dict
            render: if False, do not create a pygame display (useful for training)
        """
        self.render = bool(render)

        # Initialize pygame only when rendering is required
        if self.render:
            pygame.init()

        self.width = 800
        self.height = 600
        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird Co-Creation Sandbox")

        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Load and normalize configuration
        # Accept None or partial config; merge with config.json and defaults
        cfg = config.copy() if isinstance(config, dict) else {}
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                for k, v in file_cfg.items():
                    cfg.setdefault(k, v)
        except Exception:
            pass

        # sensible defaults
        defaults = {
            "gravity": 0.5,
            "jump_strength": -10.0,
            "pipe_speed": 3,
            "pipe_gap": 200,
            "pipe_frequency": 90,
            "background_color": [135, 206, 235],
            "bird_color": [255, 255, 0],
            "pipe_color": [34, 139, 34],
            "difficulty": "medium",
            "sound_volume": 0.8,
            "ai_player": False,
            "ai_model_path": "models/flappy_ppo",
            "bc_model_path": "models/bc_policy.pth",
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)

        self.config = cfg
        # Core gameplay parameters
        self.gravity = float(self.config.get('gravity'))
        self.jump_strength = float(self.config.get('jump_strength'))
        self.pipe_speed = float(self.config.get('pipe_speed'))
        self.pipe_gap = int(self.config.get('pipe_gap'))
        self.pipe_frequency = int(self.config.get('pipe_frequency'))

        # Colors
        self.bg_color = tuple(self.config.get('background_color'))
        self.bird_color = tuple(self.config.get('bird_color'))
        self.pipe_color = tuple(self.config.get('pipe_color'))
        
        # Game objects
        self.bird = Bird(100, self.height // 2, self.bird_color, self.jump_strength)
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        # Start paused so user can press SPACE to begin
        self.paused = True

        # Collision grace period after unpausing to avoid immediate game-over
        self.collision_enabled = False
        self.collision_timer = 0.0
        self.invuln_seconds = 0.5  # seconds of invulnerability after start

        # --- Audio initialization and sound loading ---
        # Attempt to init mixer; fail gracefully if audio not available
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._mixer_available = True
        except Exception as e:
            print(f"⚠️  Audio init failed: {e}")
            self._mixer_available = False

        # Provide paths in config under key 'sounds', e.g.:
        # "sounds": {"jump": "assets/sounds/jump.wav", "bg_music": "assets/sounds/bg_loop.ogg", "hit": "assets/sounds/hit.wav"}
        sounds_cfg = self.config.get('sounds', {}) if isinstance(self.config, dict) else {}

        def _resolve_path(p):
            if not p:
                return None
            if os.path.isabs(p):
                return p
            return os.path.join(os.path.dirname(__file__), p)

        def _load_fx(path):
            if not self._mixer_available or not path:
                return None
            full = _resolve_path(path)
            if not os.path.exists(full):
                return None
            try:
                return pygame.mixer.Sound(full)
            except Exception as e:
                print(f"⚠️  Failed to load sound {full}: {e}")
                return None

        # Short effects
        self.sounds = {}
        self.sounds['jump'] = _load_fx(sounds_cfg.get('jump') or self.config.get('jump_sound'))
        self.sounds['hit'] = _load_fx(sounds_cfg.get('hit') or self.config.get('hit_sound'))
        # Optional extra effects
        self.sounds['score'] = _load_fx(sounds_cfg.get('score') or self.config.get('score_sound') or self.config.get('point_sound'))
        self.sounds['swoosh'] = _load_fx(sounds_cfg.get('swoosh') or self.config.get('swoosh_sound') or self.config.get('menu_sound'))

        # Background music path (streaming)
        self.bg_music_path = _resolve_path(sounds_cfg.get('bg_music') or self.config.get('bg_music'))
        if self.bg_music_path and not os.path.exists(self.bg_music_path):
            self.bg_music_path = None

        # Volume (0.0 - 1.0)
        try:
            self.sound_volume = float(self.config.get('sound_volume', 0.8))
        except Exception:
            self.sound_volume = 0.8
        if self._mixer_available:
            try:
                for fx in (self.sounds.get('jump'), self.sounds.get('hit'), self.sounds.get('score'), self.sounds.get('swoosh')):
                    if fx:
                        fx.set_volume(self.sound_volume)
                pygame.mixer.music.set_volume(self.sound_volume * 0.8)
            except Exception:
                pass
        
        # Font (only if rendering)
        if self.render:
            self.font_large = pygame.font.Font(None, 72)
            self.font_medium = pygame.font.Font(None, 48)
            self.font_small = pygame.font.Font(None, 28)
        else:
            self.font_large = None
            self.font_medium = None
            self.font_small = None

        # AI player (load trained PPO if requested)
        self.ai_player = bool(self.config.get('ai_player', False))
        self.ai_model_path = self.config.get('ai_model_path', 'models/flappy_ppo')
        self._ai_model = None
        if self.ai_player:
            try:
                from stable_baselines3 import PPO
                self._ai_model = PPO.load(self.ai_model_path)
                print(f"✅ Loaded AI model from {self.ai_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load AI model: {e}")
        
        # Recording (for imitation learning): CSV of obs + action
        self._recording = False
        self._record_file = None
        self._record_writer = None
        self._last_human_action = 0
        # BC model
        self._bc_model = None
        self._bc_mean = None
        self._bc_std = None
        self._use_bc = False
        
    def spawn_pipe(self):
        """Create a new pipe."""
        pipe = Pipe(self.width, self.pipe_gap, self.pipe_color, 
                   self.pipe_speed, self.height)
        self.pipes.append(pipe)
        
    def draw_background(self):
        """Draw sky and ground with gradient effect."""
        # Sky
        self.screen.fill(self.bg_color)
        
        # Ground
        ground_height = 50
        ground_color = tuple(max(c - 40, 0) for c in self.bg_color)
        pygame.draw.rect(self.screen, ground_color, 
                        (0, self.height - ground_height, self.width, ground_height))
        
        # Ground stripes for texture
        stripe_color = tuple(max(c - 20, 0) for c in ground_color)
        for i in range(0, self.width, 40):
            pygame.draw.rect(self.screen, stripe_color,
                           (i, self.height - ground_height, 20, ground_height))
        
        # Clouds (decorative)
        cloud_color = tuple(min(c + 40, 255) for c in self.bg_color)
        for i in range(3):
            x = (self.frame_count + i * 250) % (self.width + 100) - 50
            y = 50 + i * 60
            pygame.draw.ellipse(self.screen, cloud_color, (x, y, 80, 40))
            pygame.draw.ellipse(self.screen, cloud_color, (x + 30, y - 10, 60, 40))
            pygame.draw.ellipse(self.screen, cloud_color, (x + 50, y, 70, 35))
        
    def draw_ui(self):
        """Draw score and instructions."""
        # Score with shadow effect
        score_text = self.font_large.render(str(self.score), True, (255, 255, 255))
        score_shadow = self.font_large.render(str(self.score), True, (0, 0, 0))
        
        x = self.width // 2 - score_text.get_width() // 2
        self.screen.blit(score_shadow, (x + 3, 53))
        self.screen.blit(score_text, (x, 50))
        
        # Instructions
        inst_text = self.font_small.render("SPACE: Jump | ESC: Menu", True, (255, 255, 255))
        inst_shadow = self.font_small.render("SPACE: Jump | ESC: Menu", True, (0, 0, 0))
        self.screen.blit(inst_shadow, (11, 11))
        self.screen.blit(inst_text, (10, 10))
        
        # Difficulty indicator
        diff_text = self.font_small.render(f"Difficulty: {self.config['difficulty'].upper()}", 
                                          True, (255, 255, 255))
        diff_shadow = self.font_small.render(f"Difficulty: {self.config['difficulty'].upper()}", 
                                            True, (0, 0, 0))
        self.screen.blit(diff_shadow, (self.width - diff_text.get_width() - 9, 11))
        self.screen.blit(diff_text, (self.width - diff_text.get_width() - 10, 10))
        
    def check_collisions(self):
        """Check for bird collisions with pipes or boundaries."""
        # Ground collision
        if self.bird.y + self.bird.height >= self.height - 50:
            return True
            
        # Ceiling collision
        if self.bird.y <= 0:
            return True
            
        # Pipe collisions
        for pipe in self.pipes:
            if pipe.collides_with(self.bird):
                return True
                
        return False

    # --- Audio helper methods ---
    def play_jump_sound(self):
        """Play jump effect if loaded."""
        if self._mixer_available and self.sounds.get('jump'):
            try:
                self.sounds['jump'].play()
            except Exception:
                pass

    def play_hit_sound(self):
        """Play hit/collision effect if loaded."""
        if self._mixer_available and self.sounds.get('hit'):
            try:
                self.sounds['hit'].play()
            except Exception:
                pass

    def play_score_sound(self):
        """Play score/point sound if loaded."""
        if self._mixer_available and self.sounds.get('score'):
            try:
                self.sounds['score'].play()
            except Exception:
                pass

    # --- Training / headless support ---
    def reset_for_training(self):
        """Reset the game to initial state for training (no rendering)."""
        # Reset bird and pipes and score
        self.bird.x = 100
        self.bird.y = self.height // 2
        self.bird.velocity = 0.0
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        # Spawn initial pipe a bit offscreen to start
        self.spawn_pipe()
        return self.get_observation()

    def bird_jump_for_training(self):
        """Apply jump during training (no rendering side-effects)."""
        self.bird.jump()

    def step_for_training(self):
        """Advance the game by one physics step for training.

        Returns (done: bool, reward: float)
        """
        self.frame_count += 1

        # Spawn pipes periodically
        if self.frame_count % self.pipe_frequency == 0:
            self.spawn_pipe()

        # Update bird
        self.bird.update(self.gravity)

        # Update pipes
        for pipe in self.pipes:
            pipe.update()

        # Compute reward and check scoring
        reward = -0.01  # small time penalty to encourage progress
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + pipe.width < self.bird.x:
                pipe.scored = True
                self.score += 1
                reward += 1.0

        # Remove offscreen pipes
        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

        # Check collisions
        done = False
        if self.check_collisions():
            done = True
            reward -= 5.0

        return done, reward

    def get_observation(self):
        """Return normalized observation vector for RL agent.

        Observation: [bird_y_norm, bird_vel_norm, pipe_dx_norm, gap_center_norm, pipe_gap_norm]
        All values are in range 0..1 (approx).
        """
        import numpy as np

        # Bird normalized y (0 top, 1 bottom)
        bird_y = float(self.bird.y) / max(1.0, float(self.height))
        # Bird velocity normalized (heuristic scaling)
        bird_v = (self.bird.velocity + 30.0) / 60.0
        # Find next pipe
        next_pipe = None
        for p in self.pipes:
            if p.x + p.width > self.bird.x:
                next_pipe = p
                break
        if next_pipe is None:
            # If no pipe, create a virtual far pipe
            pipe_dx = 1.0
            gap_center = 0.5
            gap = float(self.pipe_gap) / max(1.0, float(self.height))
        else:
            pipe_dx = float(next_pipe.x - self.bird.x) / max(1.0, float(self.width))
            gap_center = float(next_pipe.gap_y + next_pipe.gap / 2.0) / max(1.0, float(self.height))
            gap = float(next_pipe.gap) / max(1.0, float(self.height))

        obs = np.array([bird_y, bird_v, pipe_dx, gap_center, gap], dtype=np.float32)
        return obs

    def start_bg_music(self, loop=True):
        """Start background music (streaming)."""
        if not self._mixer_available or not self.bg_music_path:
            return
        try:
            pygame.mixer.music.load(self.bg_music_path)
            pygame.mixer.music.play(-1 if loop else 0)
        except Exception as e:
            print(f"⚠️  Failed to play bg music: {e}")

    def stop_bg_music(self):
        if self._mixer_available:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    # --- AI model runtime helpers ---
    def load_ai_model(self, path=None):
        """Load a stable-baselines3 model at runtime. Returns True on success."""
        try:
            from stable_baselines3 import PPO
        except Exception:
            print("⚠️  stable-baselines3 not available. Install with: pip install stable-baselines3[extra]")
            return False

        p = path or self.config.get('ai_model_path', 'models/flappy_ppo')
        try:
            self._ai_model = PPO.load(p)
            self.ai_player = True
            self.ai_model_path = p
            print(f"✅ AI model loaded from {p}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load AI model '{p}': {e}")
            self._ai_model = None
            return False

    def toggle_ai(self, enabled=None):
        """Enable/disable AI control. If enabled omitted, toggles current state."""
        if enabled is None:
            self.ai_player = not bool(self.ai_player)
        else:
            self.ai_player = bool(enabled)
        print(f"AI enabled = {self.ai_player}")
        
    def load_bc_model(self, model_path=None, meta_path=None):
        """Load a behavior-cloning PyTorch model and optional normalization metadata.

        model_path: path to torch model (.pth)
        meta_path: path to npz file with 'mean' and 'std' arrays
        """
        try:
            import torch
            import numpy as _np
            from torch import nn
        except Exception:
            print("⚠️  torch or numpy not available for BC model loading")
            return False

        p = model_path or self.config.get('bc_model_path') or 'models/bc_policy.pth'
        meta = meta_path or os.path.splitext(p)[0] + '_meta.npz'

        # Define BC model architecture that matches training (used if state_dict saved)
        class _BCModel(nn.Module):
            def __init__(self, input_dim=5, hidden=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden//2),
                    nn.ReLU(),
                    nn.Linear(hidden//2, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        # Attempt several loading strategies for robustness
        try:
            # First try: load as state_dict or weights-only
            state = torch.load(p, map_location='cpu')
        except RuntimeError as e:
            # PyTorch 2.6+ may enforce safe loading when full objects were saved.
            # Try to allowlist our local _BCModel class for safe unpickling.
            try:
                from torch.serialization import safe_globals
                with safe_globals([_BCModel]):
                    state = torch.load(p, map_location='cpu', weights_only=False)
            except Exception as e2:
                print(f"⚠️  BC load failed (safe unpickle attempt): {e2}")
                return False
        except Exception as e:
            print(f"⚠️  BC load failed: {e}")
            return False

        # If the loaded object is an nn.Module, use it directly
        try:
            if isinstance(state, nn.Module):
                model = state
                model.eval()
                self._bc_model = model
            elif isinstance(state, dict):
                # Could be a state_dict -> create model and load
                # Determine input dim from meta if present
                input_dim = 5
                if os.path.exists(meta):
                    try:
                        d = _np.load(meta)
                        self._bc_mean = d.get('mean')
                        self._bc_std = d.get('std')
                        if self._bc_mean is not None:
                            input_dim = int(self._bc_mean.shape[-1])
                    except Exception:
                        self._bc_mean = None
                        self._bc_std = None
                model = _BCModel(input_dim=input_dim)
                # If the dict looks like a full checkpoint with 'model_state_dict' key, prefer that
                if 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    # assume it's a state_dict for the model
                    try:
                        model.load_state_dict(state)
                    except Exception:
                        # fallback: try keys with 'net.' prefix
                        sd = {k.replace('net.', ''): v for k, v in state.items()}
                        model.load_state_dict(sd)
                model.eval()
                self._bc_model = model
            else:
                # Unknown object; try to use it directly (less common)
                self._bc_model = state
        except Exception as e:
            print(f"⚠️  Failed to convert loaded BC object into model: {e}")
            self._bc_model = None
            return False

        # Load meta if not loaded yet
        if (self._bc_mean is None or self._bc_std is None) and os.path.exists(meta):
            try:
                d = _np.load(meta)
                self._bc_mean = d.get('mean')
                self._bc_std = d.get('std')
            except Exception:
                self._bc_mean = None
                self._bc_std = None

        print(f"✅ Loaded BC model from {p}")
        return True

    def toggle_bc(self, enabled=None):
        """Enable/disable using the BC model for control."""
        if enabled is None:
            self._use_bc = not bool(self._use_bc)
        else:
            self._use_bc = bool(enabled)
        print(f"BC enabled = {self._use_bc}")
        
    def update_score(self):
        """Update score when bird passes pipes."""
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + pipe.width < self.bird.x:
                pipe.scored = True
                self.score += 1
                # play score sound if available
                try:
                    self.play_score_sound()
                except Exception:
                    pass

    # --- Recording helpers for imitation learning ---
    def start_recording(self, out_dir=None):
        """Begin recording observations and human actions to CSV.

        Creates a file data/record_<timestamp>.csv with rows: obs0,obs1,...,obs4,action
        """
        import csv
        from datetime import datetime
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(out_dir, exist_ok=True)
        fname = f"record_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
        path = os.path.join(out_dir, fname)
        try:
            f = open(path, 'w', newline='')
            w = csv.writer(f)
            # header
            header = [f'obs{i}' for i in range(5)] + ['action']
            w.writerow(header)
            self._recording = True
            self._record_file = f
            self._record_writer = w
            print(f"📁 Recording gameplay to {path}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording and close file handle."""
        if not self._recording:
            return
        try:
            self._recording = False
            if self._record_file:
                self._record_file.close()
            print("📁 Recording stopped")
        except Exception as e:
            print(f"⚠️  Error while stopping recording: {e}")
                
    def run(self):
        """Main game loop. Returns final score."""
        running = True

        # Prepare pause/prompt text
        try:
            pause_text = self.font_medium.render("Press SPACE to start (P to toggle)", True, (255, 255, 255))
        except Exception:
            pause_text = None

        # Start background music immediately when the game run begins (if configured)
        try:
            self.start_bg_music()
        except Exception:
            pass

        while running:
            # Use tick to control frame rate and get time info
            dt_ms = self.clock.tick(self.fps)
            dt = dt_ms / 1000.0
            self.frame_count += 1

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.paused:
                            # Unpause the game; start invulnerability timer
                            self.paused = False
                            self.collision_enabled = False
                            self.collision_timer = 0.0
                            # reset frame counter for nicer visuals
                            self.frame_count = 0
                            # start background music when game begins
                            try:
                                self.start_bg_music()
                            except Exception:
                                pass
                        else:
                            # Normal jump while running
                            self.bird.jump()
                            # Record that a human jump happened this frame
                            self._last_human_action = 1
                            try:
                                self.play_jump_sound()
                            except Exception:
                                pass
                    elif event.key == pygame.K_p:
                        # Toggle pause during gameplay
                        self.paused = not self.paused
                    elif event.key == pygame.K_o:
                        # toggle recording with O
                        if not self._recording:
                            self.start_recording()
                        else:
                            self.stop_recording()
                    elif event.key == pygame.K_l:
                        # Load AI model at runtime
                        try:
                            self.load_ai_model()
                        except Exception as e:
                            print(f"⚠️  Could not load AI model: {e}")
                    elif event.key == pygame.K_b:
                        # Load BC model at runtime and enable
                        try:
                            ok = self.load_bc_model()
                            if ok:
                                self.toggle_bc(True)
                        except Exception as e:
                            print(f"⚠️  Could not load BC model: {e}")
                    elif event.key == pygame.K_a:
                        # Toggle AI control on/off
                        try:
                            self.toggle_ai()
                        except Exception as e:
                            print(f"⚠️  Could not toggle AI: {e}")
                    elif event.key == pygame.K_ESCAPE:
                        try:
                            self.stop_bg_music()
                        except Exception:
                            pass
                        return self.score

            # While paused: draw a static frame and skip physics/collisions
            if self.paused:
                # Render paused screen
                self.draw_background()
                for pipe in self.pipes:
                    pipe.draw(self.screen)
                self.bird.draw(self.screen)
                self.draw_ui()
                if pause_text:
                    rect = pause_text.get_rect(center=(self.width // 2, 40))
                    self.screen.blit(pause_text, rect)
                pygame.display.flip()
                # reset last human action when paused
                self._last_human_action = 0
                continue

            # If unpaused, advance invulnerability timer until enabled
            if not self.collision_enabled:
                self.collision_timer += dt
                if self.collision_timer >= self.invuln_seconds:
                    self.collision_enabled = True

            # Spawn pipes periodically
            if self.frame_count % self.pipe_frequency == 0:
                self.spawn_pipe()

            # Update bird
            self.bird.update(self.gravity)

            # Update pipes
            for pipe in self.pipes:
                pipe.update()

            # Remove offscreen pipes
            self.pipes = [p for p in self.pipes if not p.is_offscreen()]

            # AI control (BC -> PPO)
            ai_did_jump = False
            if self._use_bc and self._bc_model is not None:
                try:
                    import numpy as _np
                    import torch as _torch
                    obs = self.get_observation()
                    if self._bc_mean is not None and self._bc_std is not None:
                        norm = (obs - self._bc_mean) / (self._bc_std + 1e-8)
                    else:
                        norm = obs
                    logits = self._bc_model(_torch.from_numpy(norm.astype(_np.float32)))
                    # expect scalar or tensor; threshold at 0.5
                    try:
                        val = logits.detach().cpu().numpy()
                        act = int((val > 0.5).astype(int))
                    except Exception:
                        act = int(float(logits) > 0.5)
                    if act == 1:
                        self.bird.jump()
                        ai_did_jump = True
                        try:
                            self.play_jump_sound()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️  BC model inference error: {e}")
                    self._use_bc = False
            elif self.ai_player and self._ai_model is not None:
                try:
                    obs = self.get_observation()
                    action, _ = self._ai_model.predict(obs, deterministic=True)
                    if int(action) == 1:
                        self.bird.jump()
                        ai_did_jump = True
                        try:
                            self.play_jump_sound()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️  AI player error: {e}")
                    self.ai_player = False

            # Compute score and rewards
            # Update score
            self.update_score()

            # Recording: if active, write observation and human action (0/1)
            if self._recording and self._record_writer is not None:
                try:
                    obs = self.get_observation()
                    # prefer human action; if human didn't act this frame but AI did, record 0
                    action = int(self._last_human_action)
                    row = list(map(float, obs.tolist())) + [action]
                    self._record_writer.writerow(row)
                except Exception as e:
                    print(f"⚠️  Recording write error: {e}")
            # reset last human action after recording
            self._last_human_action = 0

            # Check collisions (only when enabled)
            if self.collision_enabled and self.check_collisions():
                # Play hit sound and stop music, then end game
                try:
                    self.play_hit_sound()
                except Exception:
                    pass
                try:
                    self.stop_bg_music()
                except Exception:
                    pass
                running = False

            # Rendering
            self.draw_background()
            for pipe in self.pipes:
                pipe.draw(self.screen)
            self.bird.draw(self.screen)
            self.draw_ui()
            pygame.display.flip()

        return self.score

# Test the game engine directly
if __name__ == "__main__":
    test_config = {
        "gravity": 0.5,
        "jump_strength": -10,
        "pipe_speed": 3,
        "pipe_gap": 200,
        "pipe_frequency": 90,
        "background_color": [135, 206, 235],
        "bird_color": [255, 255, 0],
        "pipe_color": [34, 139, 34],
        "difficulty": "medium"
    }
    
    game = FlappyGame(test_config)
    final_score = game.run()
    print(f"Game Over! Score: {final_score}")