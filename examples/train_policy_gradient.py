"""Tabular Policy Gradient (REINFORCE) using the Tetris Gymnasium implementation.

The full Tetris state space is too large for exact tabular methods, so this
example builds a tiny placement-level MDP on top of ``tetris_gymnasium.envs.Tetris``:

- board size is reduced, defaulting to 4x4
- the piece set is reduced to O and I
- an action is a rotation plus a playable-board x position
- after placement, the next piece is sampled uniformly from the reduced set

This version uses a Monte-Carlo Policy Gradient algorithm instead of exact
Policy Iteration. A parameter matrix (Theta) maps states to action preferences,
updated via sampled episode rollouts.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris


BoardKey = tuple[int, ...]
State = tuple[BoardKey, int]
Action = tuple[int, int]


class Transition(NamedTuple):
    probability: float
    next_state: int | None
    reward: float
    done: bool


@dataclass(frozen=True)
class PolicyGradientConfig:
    width: int = 5
    height: int = 5
    gamma: float = 0.95
    learning_rate: float = 0.05
    train_episodes: int = 5000
    game_over_penalty: float = -1.0
    illegal_action_penalty: float = -1.0
    rollout_episodes: int = 20
    rollout_max_steps: int = 200


PIECE_NAMES = ("O", "I")


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


def make_model_env(config: PolicyGradientConfig) -> Tetris:
    return Tetris(
        width=config.width,
        height=config.height,
        gravity=False,
        tetrominoes=make_reduced_tetrominoes(),
    )


def empty_board_key(config: PolicyGradientConfig) -> BoardKey:
    return tuple(0 for _ in range(config.width * config.height))


def board_to_key(env: Tetris, board: np.ndarray) -> BoardKey:
    playable = env.crop_padding(board)
    return tuple(int(value > 0) for value in playable.reshape(-1))


def key_to_board(env: Tetris, board_key: BoardKey) -> np.ndarray:
    board = env.create_board()
    playable = np.array(board_key, dtype=np.uint8).reshape(env.height, env.width)
    fill_value = env.tetrominoes[0].id
    board[0 : -env.padding, env.padding : -env.padding] = playable * fill_value
    return board


def all_actions(config: PolicyGradientConfig) -> tuple[Action, ...]:
    return tuple((rotation, x) for rotation in range(4) for x in range(config.width))


def rotated_piece(env: Tetris, piece_id: int, rotation: int) -> Tetromino:
    piece = copy.copy(env.tetrominoes[piece_id])
    for _ in range(rotation % 4):
        piece = env.rotate(piece)
    return piece


def hard_drop_y(env: Tetris, piece: Tetromino, x: int) -> int | None:
    if env.collision(piece, x, 0):
        return None

    y = 0
    while not env.collision(piece, x, y + 1):
        y += 1
    return y


def legal_actions_for_state(
    env: Tetris,
    state: State,
    actions: tuple[Action, ...],
) -> list[int]:
    board_key, piece_id = state
    env.board = key_to_board(env, board_key)

    legal = []
    for action_id, (rotation, x_playable) in enumerate(actions):
        piece = rotated_piece(env, piece_id, rotation)
        x = env.padding + x_playable
        if hard_drop_y(env, piece, x) is not None:
            legal.append(action_id)
    return legal


def has_legal_placement(
    env: Tetris,
    board_key: BoardKey,
    piece_id: int,
    actions: tuple[Action, ...],
) -> bool:
    return bool(legal_actions_for_state(env, (board_key, piece_id), actions))


def transition_board_and_reward(
    env: Tetris,
    state: State,
    action: Action,
    config: PolicyGradientConfig,
) -> tuple[BoardKey | None, float, bool]:
    board_key, piece_id = state
    rotation, x_playable = action
    env.board = key_to_board(env, board_key)

    piece = rotated_piece(env, piece_id, rotation)
    x = env.padding + x_playable
    y = hard_drop_y(env, piece, x)
    if y is None:
        return None, config.illegal_action_penalty, True

    projected = env.project_tetromino(piece, x, y)
    cleared_board, lines_cleared = env.clear_filled_rows(projected.copy())
    reward = float(env.score(lines_cleared))
    return board_to_key(env, cleared_board), reward, False


def discover_reachable_states(
    env: Tetris, config: PolicyGradientConfig
) -> list[State]:
    actions = all_actions(config)
    start_states = [
        (empty_board_key(config), piece_id) for piece_id in range(len(env.tetrominoes))
    ]
    states = list(start_states)
    seen = set(start_states)
    queue = list(start_states)

    while queue:
        state = queue.pop(0)
        for action_id in legal_actions_for_state(env, state, actions):
            next_board_key, _, done = transition_board_and_reward(
                env, state, actions[action_id], config
            )
            if done:
                continue
            assert next_board_key is not None

            for next_piece_id in range(len(env.tetrominoes)):
                if not has_legal_placement(env, next_board_key, next_piece_id, actions):
                    continue
                next_state = (next_board_key, next_piece_id)
                if next_state not in seen:
                    seen.add(next_state)
                    states.append(next_state)
                    queue.append(next_state)

    return states


# --- POLICY GRADIENT SPECIFIC LOGIC ---

def get_action_probabilities(theta_s: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities for a given state's parameters, avoiding overflow."""
    shifted_logits = theta_s - np.max(theta_s)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits)


def sample_step(
    env: Tetris,
    state: State,
    action: Action,
    config: PolicyGradientConfig,
    actions: tuple[Action, ...],
    rng: np.random.Generator,
) -> tuple[State | None, float, bool]:
    """Sample a single transition from the environment."""
    next_board_key, reward, done = transition_board_and_reward(env, state, action, config)
    
    if done:
        return None, reward, True
        
    next_piece_id = rng.integers(len(PIECE_NAMES))
    if not has_legal_placement(env, next_board_key, next_piece_id, actions):
        return None, reward + config.game_over_penalty, True
        
    return (next_board_key, next_piece_id), reward, False


def train_policy_gradient(
    env: Tetris,
    states: list[State],
    actions: tuple[Action, ...],
    config: PolicyGradientConfig,
    seed: int,
) -> np.ndarray:
    """Run Tabular REINFORCE (Monte-Carlo Policy Gradient)."""
    rng = np.random.default_rng(seed)
    state_to_index = {state: index for index, state in enumerate(states)}
    
    # theta maps [state_index, action_index] to a raw preference score (logit)
    theta = np.zeros((len(states), len(actions)), dtype=np.float64)
    
    for episode in range(1, config.train_episodes + 1):
        # 1. Generate Episode
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        piece_id = rng.integers(len(PIECE_NAMES))
        current_state = (empty_board_key(config), piece_id)
        
        for step in range(config.rollout_max_steps):
            state_idx = state_to_index[current_state]
            probs = get_action_probabilities(theta[state_idx])
            action_idx = rng.choice(len(actions), p=probs)
            
            next_state, reward, done = sample_step(
                env, current_state, actions[action_idx], config, actions, rng
            )
            
            episode_states.append(state_idx)
            episode_actions.append(action_idx)
            episode_rewards.append(reward)
            
            if done or next_state is None:
                break
                
            current_state = next_state

        # 2. Calculate Returns G_t
        returns = []
        G = 0.0
        for r in reversed(episode_rewards):
            G = r + config.gamma * G
            returns.insert(0, G)
            
        # 3. Update Policy Parameters (Theta)
        for t in range(len(episode_states)):
            s_t = episode_states[t]
            a_t = episode_actions[t]
            G_t = returns[t]
            
            probs = get_action_probabilities(theta[s_t])
            
            # Gradient of ln(pi(A|S)) for softmax is: 1 - p(a_t) for a_t, and -p(a) for other actions
            grad = -probs
            grad[a_t] += 1.0
            
            # REINFORCE update step
            theta[s_t] += config.learning_rate * (config.gamma ** t) * G_t * grad
            
        if episode % max(1, config.train_episodes // 10) == 0:
            print(f"Training... Episode {episode}/{config.train_episodes} | Return: {sum(episode_rewards):.2f}")

    return theta


# ----------------------------------------


def render_board_array(board: np.ndarray) -> str:
    lines = []
    for row in board:
        lines.append("".join("." if value == 0 else str(int(value)) for value in row))
    return "\n".join(lines)


def render_board(
    env: Tetris,
    board_key: BoardKey,
    piece_id: int | None = None,
    action: Action | None = None,
) -> str:
    env.board = key_to_board(env, board_key)
    board = env.board

    if piece_id is not None and action is not None:
        rotation, x_playable = action
        piece = rotated_piece(env, piece_id, rotation)
        x = env.padding + x_playable
        y = hard_drop_y(env, piece, x)
        if y is not None:
            board = env.project_tetromino(piece, x, y)

    return render_board_array(env.crop_padding(board))


def rollout_evaluation(
    states: list[State],
    policy: list[int],
    actions: tuple[Action, ...],
    config: PolicyGradientConfig,
    episodes: int,
    max_steps: int,
    seed: int,
    render_first_episode: bool,
    render_mode: str,
    delay: float,
) -> list[float]:
    """Roll out the greedy policy derived from trained theta."""
    rng = random.Random(seed)
    state_to_index = {state: index for index, state in enumerate(states)}
    returns = []

    render_env = make_model_env(config)
    render_env.render_mode = render_mode
    if render_mode == "human":
        render_env.reset(seed=seed)

    for episode in range(episodes):
        piece_id = rng.randrange(len(PIECE_NAMES))
        current_state = (empty_board_key(config), piece_id)
        episode_return = 0.0

        for step in range(max_steps):
            state_idx = state_to_index[current_state]
            board_key, active_piece_id = current_state
            
            # Follow greedy policy extracted from trained PG
            action_id = policy[state_idx]
            rotation, x = actions[action_id]

            if render_first_episode and episode == 0:
                render_env.board = key_to_board(render_env, board_key)
                render_env.active_tetromino = rotated_piece(
                    render_env, active_piece_id, 0
                )

                if render_mode == "ansi":
                    print(render_env.render() + "\n")
                elif render_mode == "human":
                    render_env.render()

            # Execute action using the original explicit transition
            next_board_key, reward, done = transition_board_and_reward(
                render_env, current_state, actions[action_id], config
            )
            episode_return += reward

            if render_first_episode and episode == 0:
                lines_cleared = int((reward / config.width) ** 0.5) if reward > 0 else 0
                print(
                    f"step={step} piece={PIECE_NAMES[active_piece_id]} "
                    f"action=(rot={rotation}, x={x}) "
                    f"reward={reward:.1f} lines={lines_cleared}"
                )

                if render_mode == "human":
                    import cv2
                    cv2.waitKey(max(1, int(delay * 1000)))
                elif delay > 0:
                    time.sleep(delay)

            if done or next_board_key is None:
                if render_first_episode and episode == 0:
                    print("Game Over!")
                break
                
            # Sample next piece to continue
            next_piece_id = rng.randrange(len(PIECE_NAMES))
            if not has_legal_placement(render_env, next_board_key, next_piece_id, actions):
                if render_first_episode and episode == 0:
                    print("No legal placements remain. Game Over!")
                break
                
            current_state = (next_board_key, next_piece_id)

        returns.append(episode_return)

    return returns


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    defaults = PolicyGradientConfig()
    parser = argparse.ArgumentParser(
        description="Run Tabular Policy Gradient (REINFORCE) on a tiny Tetris MDP."
    )
    parser.add_argument("--width", type=positive_int, default=defaults.width)
    parser.add_argument("--height", type=positive_int, default=defaults.height)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=defaults.learning_rate,
        help="Step size for the policy parameter updates."
    )
    parser.add_argument(
        "--train-episodes",
        type=positive_int,
        default=defaults.train_episodes,
        help="Number of episodes to train using REINFORCE."
    )
    parser.add_argument(
        "--game-over-penalty",
        type=float,
        default=defaults.game_over_penalty,
    )
    parser.add_argument(
        "--illegal-action-penalty",
        type=float,
        default=defaults.illegal_action_penalty,
    )
    parser.add_argument("--episodes", type=positive_int, default=defaults.rollout_episodes)
    parser.add_argument("--max-steps", type=positive_int, default=defaults.rollout_max_steps)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render-first-episode", action="store_true")
    parser.add_argument("--render-mode", choices=("ansi", "human"), default="ansi")
    parser.add_argument("--delay", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyGradientConfig(
        width=args.width,
        height=args.height,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        train_episodes=args.train_episodes,
        game_over_penalty=args.game_over_penalty,
        illegal_action_penalty=args.illegal_action_penalty,
    )
    env = make_model_env(config)

    print("Discovering reachable states...")
    states = discover_reachable_states(env, config)
    actions = all_actions(config)
    
    print(f"Found {len(states)} states. Beginning Policy Gradient Training...")
    theta = train_policy_gradient(env, states, actions, config, args.seed)
    
    # Extract greedy deterministic policy for evaluation
    greedy_policy = np.argmax(theta, axis=1).tolist()

    returns = rollout_evaluation(
        states=states,
        policy=greedy_policy,
        actions=actions,
        config=config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render_first_episode=args.render_first_episode,
        render_mode=args.render_mode,
        delay=args.delay,
    )

    mean_return = sum(returns) / len(returns)

    print()
    print("Policy Gradient (REINFORCE) training finished")
    print(f"board={config.width}x{config.height} pieces={','.join(PIECE_NAMES)}")
    print(f"states={len(states)} actions={len(actions)} train_episodes={config.train_episodes}")
    print(
        f"rollout_mean_return={mean_return:.3f} "
        f"min={min(returns):.3f} max={max(returns):.3f}"
    )


if __name__ == "__main__":
    main()