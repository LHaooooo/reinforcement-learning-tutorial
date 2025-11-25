#!/usr/bin/env python3
"""Quick render smoke test for Gym/Gymnasium environments.

Usage:
  python scripts/test_gym_render.py            # default CartPole-v1, 10 seconds
  python scripts/test_gym_render.py --env Pendulum-v1 --seconds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

PRINT_PREFIX = "[gym-render]"


def _import_gym() -> Any:
    """Import gymnasium if available, otherwise fall back to legacy gym."""
    try:
        import gymnasium as gym  # type: ignore

        return gym
    except ImportError:
        import gym  # type: ignore

        return gym


def run(env_id: str, seconds: float, fps: int, verbose: bool) -> None:
    gym = _import_gym()
    render_mode = "human"

    try:
        env = gym.make(env_id, render_mode=render_mode)
    except TypeError:
        # Older gym API without render_mode; fall back to default constructor.
        env = gym.make(env_id)
        render_mode = None

    obs, info = env.reset() if hasattr(env, "reset") else (env.reset(), {})
    frame_time = 1.0 / fps
    end_time = time.time() + seconds

    print(
        f"{PRINT_PREFIX} Using {gym.__name__} {getattr(gym, '__version__', 'unknown')}",
        flush=True,
    )
    print(
        f"{PRINT_PREFIX} Environment: {env_id}, render_mode={render_mode or 'legacy'}",
        flush=True,
    )
    print(
        f"{PRINT_PREFIX} Running for ~{seconds:.1f}s at {fps} FPS. Close the window or Ctrl+C to stop.",
        flush=True,
    )

    last_log = time.time()
    try:
        while time.time() < end_time:
            action = env.action_space.sample()
            step_out = env.step(action)

            if len(step_out) == 4:
                obs, reward, done, info = step_out
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = step_out

            if render_mode:
                env.render()

            if terminated or truncated:
                obs, info = env.reset() if hasattr(env, "reset") else (env.reset(), {})

            if verbose and time.time() - last_log >= 1.0:
                last_log = time.time()
                print(
                    f"{PRINT_PREFIX} t={last_log:.0f}s reward={reward:.3f} terminated={terminated} truncated={truncated}",
                    flush=True,
                )

            time.sleep(frame_time)
    except KeyboardInterrupt:
        print(f"\n{PRINT_PREFIX} Interrupted by user.", flush=True)
    finally:
        env.close()
        print(
            f"{PRINT_PREFIX} Render test finished and environment closed.", flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render smoke test for Gym/Gymnasium.")
    parser.add_argument("--env", default="CartPole-v1", help="Environment ID to test.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Duration to run.")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print step info once per second."
    )
    args = parser.parse_args()

    run(env_id=args.env, seconds=args.seconds, fps=args.fps, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
