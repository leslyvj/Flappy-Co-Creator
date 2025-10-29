import os
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.flappy_env import FlappyEnv


def main(total_timesteps: int = 200_000, model_out: str = "models/flappy_ppo"):
    # Ensure output directory exists
    out_dir = os.path.dirname(model_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Starting training: timesteps={total_timesteps}, model_out={model_out}")
    env = DummyVecEnv([lambda: FlappyEnv()])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, tensorboard_log="./tb/")

    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"⚠️  Training failed: {e}")
        # attempt to save partial model if available
        try:
            ts = int(time.time())
            fallback = f"{model_out}_failed_{ts}"
            model.save(fallback)
            print(f"Saved fallback model to {fallback}")
        except Exception as e2:
            print(f"⚠️  Could not save fallback model: {e2}")
        raise

    try:
        model.save(model_out)
        print(f"✅ Model saved to {model_out}")
    except Exception as e:
        print(f"⚠️  Failed to save model to {model_out}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000, help='Number of training timesteps')
    parser.add_argument('--out', type=str, default='models/flappy_ppo', help='Output model path prefix')
    args = parser.parse_args()
    main(total_timesteps=args.timesteps, model_out=args.out)
