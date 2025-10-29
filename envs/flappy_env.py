import gym
import numpy as np
from gym import spaces

from game_engine import FlappyGame


class FlappyEnv(gym.Env):
    """Gym environment wrapper around FlappyGame for RL training.

    Observation space (5): [bird_y, bird_v, pipe_dx, gap_center, gap]
    Action space: Discrete(2) => 0 = no-op, 1 = flap
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        # Create headless game (no rendering)
        self.game = FlappyGame(self.config, render=False)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        _ = super().reset(seed=seed)
        obs = self.game.reset_for_training()
        return obs, {}

    def step(self, action):
        # action: 0 or 1
        if int(action) == 1:
            self.game.bird_jump_for_training()
        done, reward = self.game.step_for_training()
        obs = self.game.get_observation()
        return obs, reward, done, False, {}

    def render(self):
        # Optional: create a render-capable FlappyGame in render mode and copy state
        pass

    def close(self):
        pass
