"""Random placement agent for a reduced Tetris Gymnasium board.

This uses the real ``tetris_gymnasium.envs.Tetris`` implementation with a small
board and a reduced piece set.  The agent samples a legal placement uniformly at
random, applies the chosen rotation/x-position, and hard-drops the piece.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris


Action = tuple[int, int]


def make_reduced_tetrominoes() -> list[Tetromino]:
    """Return fresh, un-offset pieces for Tetris.__init__ to preprocess."""
    return [
        Tetromino(
            0, 
            [0, 0, 240],  # Jミノの一般的な色（青）
            np.array(
                [
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                ],
                dtype=np.uint8,
            )
        ),
        Tetromino(
            1,
            [128, 0, 128],
            np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ),
    ]


def make_env(width: int, height: int, render_mode: str) -> Tetris:
    return Tetris(
        render_mode=render_mode,
        width=width,
        height=height,
        gravity=False,
        tetrominoes=make_reduced_tetrominoes(),
    )


def rotated_piece(env: Tetris, rotation: int) -> Tetromino:
    piece = copy.copy(env.active_tetromino)
    for _ in range(rotation % 4):
        piece = env.rotate(piece)
    return piece


def can_hard_drop(env: Tetris, piece: Tetromino, x: int) -> bool:
    if env.collision(piece, x, 0):
        return False

    y = 0
    while not env.collision(piece, x, y + 1):
        y += 1
    return True


def legal_placements(env: Tetris) -> list[Action]:
    placements = []
    for rotation in range(4):
        piece = rotated_piece(env, rotation)
        for x_playable in range(env.width):
            x = env.padding + x_playable
            if can_hard_drop(env, piece, x):
                placements.append((rotation, x_playable))
    return placements


def apply_placement(env: Tetris, placement: Action):
    rotation, x_playable = placement
    env.active_tetromino = rotated_piece(env, rotation)
    env.x = env.padding + x_playable
    env.y = 0
    return env.step(env.actions.hard_drop)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play reduced Tetris with a random legal-placement agent."
    )
    parser.add_argument("--width", type=positive_int, default=5)
    parser.add_argument("--height", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=positive_int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--render-mode", choices=("ansi", "human"), default="ansi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    env = make_env(args.width, args.height, args.render_mode)
    env.reset(seed=args.seed)

    total_reward = 0.0
    terminated = False

    for step in range(args.max_steps):
        if args.render_mode == "ansi":
            print(env.render() + "\n")
        else:
            env.render()

        placements = legal_placements(env)
        if not placements:
            print("No legal placements remain.")
            break

        placement = rng.choice(placements)
        _, reward, terminated, truncated, info = apply_placement(env, placement)
        total_reward += reward

        rotation, x = placement
        print(
            f"step={step} action=(rot={rotation}, x={x}) "
            f"reward={reward} lines={info['lines_cleared']}"
        )

        if terminated or truncated:
            break
        if args.render_mode == "human":
            import cv2

            cv2.waitKey(max(1, int(args.delay * 1000)))
        elif args.delay > 0:
            time.sleep(args.delay)

    print(f"Game Over! total_reward={total_reward}")


if __name__ == "__main__":
    main()
