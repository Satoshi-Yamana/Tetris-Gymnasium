"""Tabular policy iteration using the Tetris Gymnasium implementation."""

from __future__ import annotations

import argparse
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

# --- 共通ユーティリティからのインポート ---
from small_tetris_utils import rotated_piece, hard_drop_y


BoardKey = tuple[int, ...]
State = tuple[BoardKey, int]
Action = tuple[int, int]


class Transition(NamedTuple):
    probability: float
    next_state: int | None
    reward: float
    done: bool


@dataclass(frozen=True)
class PolicyIterationConfig:
    width: int = 5
    height: int = 5
    gamma: float = 0.95
    theta: float = 1e-8
    max_policy_iterations: int = 100
    max_eval_iterations: int = 10_000
    game_over_penalty: float = -1.0
    illegal_action_penalty: float = -1.0
    rollout_episodes: int = 20
    rollout_max_steps: int = 200


PIECE_NAMES = ("O", "T")


def make_reduced_tetrominoes() -> list[Tetromino]:
    """Return fresh, un-offset pieces for Tetris.__init__ to preprocess."""
    return [
        Tetromino(0, [240, 240, 0], np.array([[1, 1], [1, 1]], dtype=np.uint8)),
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


def make_model_env(config: PolicyIterationConfig) -> Tetris:
    return Tetris(
        width=config.width,
        height=config.height,
        gravity=False,
        tetrominoes=make_reduced_tetrominoes(),
    )


def empty_board_key(config: PolicyIterationConfig) -> BoardKey:
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


def all_actions(config: PolicyIterationConfig) -> tuple[Action, ...]:
    return tuple((rotation, x) for rotation in range(4) for x in range(config.width))


def legal_actions_for_state(
    env: Tetris,
    state: State,
    actions: tuple[Action, ...],
) -> list[int]:
    board_key, piece_id = state
    env.board = key_to_board(env, board_key)

    legal = []
    for action_id, (rotation, x_playable) in enumerate(actions):
        # 共通関数の呼び出し
        piece = rotated_piece(env, env.tetrominoes[piece_id], rotation)
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
    config: PolicyIterationConfig,
) -> tuple[BoardKey | None, float, bool]:
    board_key, piece_id = state
    rotation, x_playable = action
    env.board = key_to_board(env, board_key)

    # 共通関数の呼び出し
    piece = rotated_piece(env, env.tetrominoes[piece_id], rotation)
    x = env.padding + x_playable
    y = hard_drop_y(env, piece, x)
    if y is None:
        return None, config.illegal_action_penalty, True

    projected = env.project_tetromino(piece, x, y)
    cleared_board, lines_cleared = env.clear_filled_rows(projected.copy())
    reward = float(env.score(lines_cleared))
    return board_to_key(env, cleared_board), reward, False


def next_state_distribution(
    env: Tetris,
    state: State,
    action_id: int,
    state_to_index: dict[State, int],
    config: PolicyIterationConfig,
    actions: tuple[Action, ...],
) -> list[Transition]:
    next_board_key, reward, done = transition_board_and_reward(
        env, state, actions[action_id], config
    )
    if done:
        return [Transition(1.0, None, reward, True)]

    assert next_board_key is not None
    probability = 1.0 / len(env.tetrominoes)
    transitions = []
    for next_piece_id in range(len(env.tetrominoes)):
        next_state = (next_board_key, next_piece_id)
        if has_legal_placement(env, next_board_key, next_piece_id, actions):
            transitions.append(
                Transition(probability, state_to_index[next_state], reward, False)
            )
        else:
            transitions.append(
                Transition(
                    probability,
                    None,
                    reward + config.game_over_penalty,
                    True,
                )
            )
    return transitions


def discover_reachable_states(
    env: Tetris, config: PolicyIterationConfig
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


def build_transition_table(
    env: Tetris,
    states: list[State],
    config: PolicyIterationConfig,
) -> tuple[list[list[list[Transition]]], tuple[Action, ...]]:
    actions = all_actions(config)
    state_to_index = {state: index for index, state in enumerate(states)}
    transitions = [
        [
            next_state_distribution(env, state, action_id, state_to_index, config, actions)
            for action_id in range(len(actions))
        ]
        for state in states
    ]
    return transitions, actions


def expected_action_value(
    transitions: list[Transition], values: list[float], gamma: float
) -> float:
    total = 0.0
    for transition in transitions:
        value = transition.reward
        if not transition.done:
            assert transition.next_state is not None
            value += gamma * values[transition.next_state]
        total += transition.probability * value
    return total


def policy_iteration(
    states: list[State],
    transitions: list[list[list[Transition]]],
    actions: tuple[Action, ...],
    config: PolicyIterationConfig,
) -> tuple[list[int], list[float], int]:
    values = [0.0 for _ in states]
    policy = [
        max(
            range(len(actions)),
            key=lambda action_id: expected_action_value(
                transitions[state_id][action_id], values, config.gamma
            ),
        )
        for state_id in range(len(states))
    ]

    for policy_iteration_id in range(1, config.max_policy_iterations + 1):
        for _ in range(config.max_eval_iterations):
            delta = 0.0
            for state_id in range(len(states)):
                old_value = values[state_id]
                values[state_id] = expected_action_value(
                    transitions[state_id][policy[state_id]], values, config.gamma
                )
                delta = max(delta, abs(old_value - values[state_id]))
            if delta < config.theta:
                break

        policy_stable = True
        for state_id in range(len(states)):
            old_action = policy[state_id]
            best_action = max(
                range(len(actions)),
                key=lambda action_id: expected_action_value(
                    transitions[state_id][action_id], values, config.gamma
                ),
            )
            policy[state_id] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            return policy, values, policy_iteration_id

    return policy, values, config.max_policy_iterations


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
        # 共通関数の呼び出し
        piece = rotated_piece(env, env.tetrominoes[piece_id], rotation)
        x = env.padding + x_playable
        y = hard_drop_y(env, piece, x)
        if y is not None:
            board = env.project_tetromino(piece, x, y)

    return render_board_array(env.crop_padding(board))


def sample_transition(transitions: list[Transition], rng: random.Random) -> Transition:
    threshold = rng.random()
    cumulative = 0.0
    for transition in transitions:
        cumulative += transition.probability
        if threshold <= cumulative:
            return transition
    return transitions[-1]


def rollout(
    states: list[State],
    transitions: list[list[list[Transition]]],
    policy: list[int],
    actions: tuple[Action, ...],
    config: PolicyIterationConfig,
    episodes: int,
    max_steps: int,
    seed: int,
    render_first_episode: bool,
    render_mode: str,
    delay: float,
) -> list[float]:
    rng = random.Random(seed)
    state_to_index = {state: index for index, state in enumerate(states)}
    returns = []

    # 指定された render_mode で描画用環境を作る
    render_env = make_model_env(config)
    render_env.render_mode = render_mode
    # humanモードの時は一度resetを呼んでウィンドウを初期化する必要がある
    if render_mode == "human":
        render_env.reset(seed=seed)

    for episode in range(episodes):
        piece_id = rng.randrange(len(PIECE_NAMES))
        state_id = state_to_index[(empty_board_key(config), piece_id)]
        episode_return = 0.0

        for step in range(max_steps):
            board_key, active_piece_id = states[state_id]
            action_id = policy[state_id]
            rotation, x = actions[action_id]

            # --- 描画処理 ---
            if render_first_episode and episode == 0:
                # 描画用環境に状態を同期
                render_env.board = key_to_board(render_env, board_key)
                # 共通関数の呼び出し
                render_env.active_tetromino = rotated_piece(
                    render_env, render_env.tetrominoes[active_piece_id], 0
                )

                # モードによって描写方法を変える
                if render_mode == "ansi":
                    print(render_env.render() + "\n")
                elif render_mode == "human":
                    render_env.render()
            # ----------------

            transition = sample_transition(transitions[state_id][action_id], rng)
            episode_return += transition.reward

            # --- ログ出力 & ウェイト処理 ---
            if render_first_episode and episode == 0:
                # 報酬がプラスの時だけライン数を逆算し、マイナス（ペナルティ）の時は0にする
                if transition.reward > 0:
                    lines_cleared = int((transition.reward / config.width) ** 0.5)
                else:
                    lines_cleared = 0

                print(
                    f"step={step} piece={PIECE_NAMES[active_piece_id]} "
                    f"action=(rot={rotation}, x={x}) "
                    f"reward={transition.reward:.1f} lines={lines_cleared}"
                )

                if render_mode == "human":
                    import cv2
                    cv2.waitKey(max(1, int(delay * 1000)))
                elif delay > 0:
                    time.sleep(delay)
            # ----------------

            if transition.done:
                if render_first_episode and episode == 0:
                    print("No legal placements remain. Game Over!")
                break
            assert transition.next_state is not None
            state_id = transition.next_state

        returns.append(episode_return)

    return returns


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    defaults = PolicyIterationConfig()
    parser = argparse.ArgumentParser(
        description="Run tabular policy iteration on a tiny Tetris Gymnasium MDP."
    )
    parser.add_argument("--width", type=positive_int, default=defaults.width)
    parser.add_argument("--height", type=positive_int, default=defaults.height)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--theta", type=float, default=defaults.theta)
    parser.add_argument(
        "--max-policy-iterations",
        type=positive_int,
        default=defaults.max_policy_iterations,
    )
    parser.add_argument(
        "--max-eval-iterations",
        type=positive_int,
        default=defaults.max_eval_iterations,
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
    config = PolicyIterationConfig(
        width=args.width,
        height=args.height,
        gamma=args.gamma,
        theta=args.theta,
        max_policy_iterations=args.max_policy_iterations,
        max_eval_iterations=args.max_eval_iterations,
    )
    env = make_model_env(config)

    states = discover_reachable_states(env, config)
    transitions, actions = build_transition_table(env, states, config)
    policy, values, iterations = policy_iteration(states, transitions, actions, config)

    returns = rollout(
        states=states,
        transitions=transitions,
        policy=policy,
        actions=actions,
        config=config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render_first_episode=args.render_first_episode,
        render_mode=args.render_mode,
        delay=args.delay,
    )

    start_values = [
        values[states.index((empty_board_key(config), piece_id))]
        for piece_id in range(len(env.tetrominoes))
    ]
    mean_return = sum(returns) / len(returns)

    print()
    print("Policy iteration finished")
    print(f"board={config.width}x{config.height} pieces={','.join(PIECE_NAMES)}")
    print("model=tetris_gymnasium.envs.Tetris")
    print(f"states={len(states)} actions={len(actions)} iterations={iterations}")
    print(
        "start_values="
        + ", ".join(
            f"{PIECE_NAMES[piece_id]}:{value:.3f}"
            for piece_id, value in enumerate(start_values)
        )
    )
    print(
        f"rollout_mean_return={mean_return:.3f} "
        f"min={min(returns):.3f} max={max(returns):.3f}"
    )


if __name__ == "__main__":
    main()
