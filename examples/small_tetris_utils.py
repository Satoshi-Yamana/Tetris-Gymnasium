"""Shared utility functions for Tetris Gymnasium MDPs."""

from __future__ import annotations

import copy

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris


Action = tuple[int, int]


def rotated_piece(env: Tetris, piece: Tetromino, rotation: int) -> Tetromino:
    """Returns a piece rotated by the specified number of 90-degree turns."""
    rotated = copy.copy(piece)
    for _ in range(rotation % 4):
        rotated = env.rotate(rotated)
    return rotated


def hard_drop_y(env: Tetris, piece: Tetromino, x: int) -> int | None:
    """Calculates the y-coordinate of a hard drop. Returns None if invalid."""
    if env.collision(piece, x, 0):
        return None

    y = 0
    while not env.collision(piece, x, y + 1):
        y += 1
    return y


def get_legal_placements(env: Tetris) -> list[Action]:
    """Returns a list of all legal (rotation, x_playable) actions for the active piece."""
    placements = []
    for rotation in range(4):
        piece = rotated_piece(env, env.active_tetromino, rotation)
        for x_playable in range(env.width):
            x = env.padding + x_playable
            if hard_drop_y(env, piece, x) is not None:
                placements.append((rotation, x_playable))
    return placements


def apply_placement(env: Tetris, placement: Action):
    """Applies a specific rotation and x-position, then executes a hard drop."""
    rotation, x_playable = placement
    env.active_tetromino = rotated_piece(env, env.active_tetromino, rotation)
    env.x = env.padding + x_playable
    env.y = 0
    return env.step(env.actions.hard_drop)