# Flappy Co-Creator

**An intelligent Flappy Bird sandbox for AI experimentation and human-AI collaboration.**

Train AI agents through imitation learning (Behavior Cloning) or reinforcement learning (PPO). Modify game parameters dynamically with natural language commands powered by local LLMs. A comprehensive framework for studying game AI and machine learning algorithms.

---

## Overview

Flappy Co-Creator provides a complete environment for:

- **Interactive Gameplay** - Classic Flappy Bird mechanics built with Pygame
- **Data Collection** - Record human gameplay sessions for supervised learning
- **Behavior Cloning** - Train neural networks to imitate human strategies
- **Reinforcement Learning** - Develop optimal policies through trial-and-error learning
- **Live Inference** - Load and evaluate trained models during gameplay
- **Dynamic Configuration** - Modify physics and game parameters via LLM integration

---

## Installation

### Core Dependencies

```bash
pip install pygame numpy torch
```

### Optional Components

**For Reinforcement Learning:**
```bash
pip install stable-baselines3[extra] gymnasium
```

**For LLM Integration:**
```bash
pip install ollama
```

**PyTorch Installation:**  
Refer to the [official PyTorch guide](https://pytorch.org/get-started/locally/) for platform-specific instructions (CPU/GPU).

---

## Quick Start

```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Jump / Start game |
| `P` | Pause/Unpause |
| `O` | Toggle gameplay recording |
| `B` | Load Behavior Cloning model |
| `L` | Load PPO model |
| `A` | Toggle AI control |
| `ESC` | Exit application |

---

## Usage Guide

### 1. Recording Human Gameplay

Collect training data by recording your play sessions:

```bash
# Launch the game
python main.py

# Press 'O' to start recording
# Play for several minutes
# Press 'O' again to stop recording
```

Recordings are saved to `data/record_<timestamp>.csv` with the following schema:

| Column | Description | Value Range |
|--------|-------------|-------------|
| obs0 | Bird Y position (normalized) | [0, 1] |
| obs1 | Bird velocity (normalized) | [-1, 1] |
| obs2 | Horizontal distance to pipe | [0, 1] |
| obs3 | Vertical distance to gap | [0, 1] |
| obs4 | Pipe gap size (normalized) | [0, 1] |
| action | Jump command | {0, 1} |

---

### 2. Training Behavior Cloning Model

Train a neural network to imitate recorded human gameplay:

```bash
python train_bc.py --data-dir data --epochs 50 --batch-size 64
```

**Output Files:**
- `bc_policy.pth` - Trained PyTorch model
- `bc_policy_meta.npz` - Normalization parameters (mean, std)

**Loading in Game:**
- Press `B` during gameplay to activate the BC model
- The model will automatically take control upon loading

---

### 3. Training Reinforcement Learning Agent

Train an agent using Proximal Policy Optimization (PPO):

```bash
# Development test
python train_agent.py --timesteps 5000

# Full training run
python train_agent.py --timesteps 1000000 --checkpoint-freq 50000
```

**Monitor Training Progress:**
```bash
tensorboard --logdir tb
```

Trained models are saved to `models/` in Stable-Baselines3 `.zip` format.

**Loading in Game:**
- Press `L` to load the PPO model
- Press `A` to toggle autonomous control

---

## Project Structure

```
flappy-co-creator/
│
├── main.py                     # Application entry point
├── game_engine.py              # Core game logic and AI integration
├── ai_co_creator.py            # LLM-based configuration generator
├── config.json                 # Global configuration file
│
├── flappy_env.py               # Gymnasium environment wrapper
├── train_agent.py              # PPO training pipeline
├── train_bc.py                 # Behavior Cloning training pipeline
│
├── assets/                     # Audio and visual resources
├── data/                       # Recorded gameplay datasets
├── models/                     # Trained model artifacts
└── tb/                         # TensorBoard logging directory
```

---

## Configuration

The `config.json` file controls game behavior, physics, and AI settings.

### Game Physics

```json
{
  "gravity": 0.5,
  "jump_strength": -10.0,
  "pipe_speed": 3,
  "pipe_gap": 150,
  "pipe_frequency": 1500
}
```

### AI Model Paths

```json
{
  "ai_player": false,
  "ai_model_path": "models/ppo_flappy_1000000_steps.zip",
  "bc_model_path": "bc_policy.pth"
}
```

### Audio Settings

```json
{
  "sounds": {
    "jump": "assets/sounds/sfx_wing.mp3",
    "score": "assets/sounds/sfx_point.mp3",
    "bg_music": "assets/sounds/music.mp3"
  },
  "sound_volume": 0.7
}
```

---

## Training Guidelines

### Behavior Cloning Best Practices

- **Data Diversity:** Record multiple sessions with varying strategies and outcomes
- **Session Duration:** 2-5 minutes per recording recommended
- **Action Balance:** Ensure adequate representation of both jump and no-jump actions
- **Training Epochs:** Start with 30-50 epochs; increase if validation loss plateaus
- **Batch Size:** 32-128 depending on dataset size and available memory

### Reinforcement Learning Recommendations

- **Training Steps:** Minimum 500,000 for basic performance; 1,000,000+ for strong agents
- **Checkpointing:** Save models every 50,000-100,000 steps
- **Hardware:** GPU acceleration significantly reduces training time
- **Reward Engineering:** Modify `flappy_env.py` if convergence is slow
- **Hyperparameters:** Default PPO settings work well; tune learning rate if needed

---

## Performance Benchmarks

| Method | Training Duration | Average Score | Notes |
|--------|------------------|---------------|-------|
| Human (Novice) | N/A | 5-15 | Baseline performance |
| BC (30 epochs) | 2-5 minutes | 10-30 | Replicates human behavior |
| PPO (500k steps) | 1-2 hours | 20-50 | Basic autonomous play |
| PPO (2M steps) | 6-8 hours | 100+ | Advanced strategies |

Performance varies based on hardware and training data quality.

---

## Troubleshooting

### Behavior Cloning Issues

**Model fails to load:**
- Verify `bc_policy.pth` and `bc_policy_meta.npz` exist
- Check file paths in `config.json`
- Ensure PyTorch version compatibility

**Poor model performance:**
- Increase training epochs
- Collect more diverse gameplay data
- Verify action distribution in training data

### Reinforcement Learning Issues

**Agent doesn't learn:**
- Confirm observation space consistency (5 features)
- Verify reward function in `flappy_env.py`
- Check TensorBoard for loss convergence

**Model loads but doesn't act:**
- Ensure observation normalization matches training
- Verify model was trained to completion
- Check for version mismatches in Stable-Baselines3

### Recording Issues

**Data not saving:**
- Confirm `data/` directory exists
- Verify write permissions
- Check console for error messages

---

## Technical Details

### Observation Space

The environment provides a 5-dimensional state vector:

1. **Bird Y Position** - Normalized vertical position [0, 1]
2. **Bird Velocity** - Normalized vertical velocity [-1, 1]
3. **Pipe Distance** - Normalized horizontal distance to next pipe [0, 1]
4. **Gap Center** - Normalized vertical distance to gap center [0, 1]
5. **Gap Size** - Normalized pipe gap dimension [0, 1]

### Action Space

Binary action space: {0: no jump, 1: jump}

### Neural Network Architecture (BC)

- Input Layer: 5 neurons (observation features)
- Hidden Layers: 2 fully-connected layers (128, 64 neurons)
- Output Layer: 1 neuron with sigmoid activation
- Loss Function: Binary Cross-Entropy with Logits
- Optimizer: Adam

---

## Requirements

**Minimum:**
- Python 3.10 or higher
- 4GB RAM
- CPU-only training supported

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (for RL training)

---

## Use Cases

- **Machine Learning Research:** Comparative study of imitation vs reinforcement learning
- **Education:** Hands-on learning environment for AI/ML concepts
- **Game Development:** Prototype and evaluate AI agent behaviors
- **Algorithm Development:** Test custom learning algorithms and reward functions

---

## System Requirements

**Operating Systems:**
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 20.04+)

**Python Versions:**
- Tested on Python 3.10, 3.11

---

## Known Limitations

- Single-agent environment only
- No support for curriculum learning
- Limited to discrete action space
- Requires manual recording for BC training data

---

## Future Development

Planned features and improvements:

- Multi-agent competitive environments
- Additional RL algorithms (DQN, A3C, SAC)
- Automated hyperparameter tuning
- Web-based training dashboard
- Pre-trained model repository

---

## License

This project is provided for educational and research purposes.

---

## Contributing

Contributions are welcome. Please submit pull requests for:

- Bug fixes and patches
- Documentation improvements
- New training algorithms
- Performance optimizations
- Test coverage

---

## Support

For issues, questions, or feature requests, please open an issue on the project repository.

---


---

**Documentation Version:** 1.0  
**Last Updated:** 2025
