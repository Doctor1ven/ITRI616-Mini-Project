# Peg Solitaire AI

A Python reinforcement-learning project for the standard English Peg Solitaire board.

The project trains a tabular Q-learning agent to play Peg Solitaire from the classic starting position: 32 pegs on a 33-hole cross-shaped board with the centre hole empty. A legal move jumps one peg over an adjacent peg into an empty hole, removing the jumped peg. The goal is to finish with exactly one peg remaining.

## Features

- Standard English Peg Solitaire board and legal move rules
- Tabular Q-learning agent with epsilon-greedy exploration
- Reverse-curriculum training from solvable near-endgame boards
- Optional exact DFS helper for small late-game positions
- Training logs saved as JSON
- Training graph generation with Matplotlib
- Saved successful solution replays
- Tkinter GUI for training, watching the agent, and replaying solutions

## Project Structure

| Path | Description |
| --- | --- |
| `agent.py` | Q-learning agent, action selection, Q-table updates, save/load support |
| `game.py` | Board representation, legal moves, reverse moves, rewards, terminal checks |
| `train.py` | Command-line training loop, curriculum training, logging, graph output |
| `solver.py` | Exact DFS helper for low-peg endgames |
| `gui.py` | Tkinter interface for visual training and watching the agent |
| `main.py` | Main GUI launcher |
| `replay.py` | GUI replay tool for saved solution JSON files |
| `utils.py` | Logging, plotting, JSON conversion, and directory helpers |
| `training_logs/` | Generated training logs and graph images |
| `saved_solutions/` | Saved successful 31-move solution files |
| `qtable.pkl` | Saved trained Q-table, if present |

## Requirements

- Python 3.10 or newer
- Tkinter for the GUI
- NumPy
- Matplotlib

Tkinter is included with most standard Python installations on Windows. On some Linux distributions it may need to be installed separately.

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone <repo-url>
cd peg-solitaire-ai
pip install -r requirements.txt
```

## Usage

Open the main GUI menu:

```bash
python main.py
```

Train from the command line:

```bash
python train.py --episodes 10000 --progress 500
```

Train from scratch instead of loading an existing Q-table:

```bash
python train.py --episodes 10000 --fresh
```

Use a custom Q-table path:

```bash
python train.py --episodes 10000 --qtable qtable.pkl
```

Disable reverse-curriculum training for comparison:

```bash
python train.py --episodes 10000 --no-curriculum
```

Replay saved successful solutions:

```bash
python replay.py
```

## Training Outputs

Training creates or updates these files:

- `qtable.pkl`: pickled Q-table used by the agent
- `training_logs/training_log_*.json`: episode metrics
- `training_logs/training_graphs_*.png`: generated performance graphs
- `saved_solutions/solution_ep*.json`: successful 31-move one-peg solutions

The Q-table can become large after long training runs. If you publish this project on GitHub, consider excluding `qtable.pkl` from normal commits or storing it with Git LFS.

## Current Results

The included training runs show that the learned policy performs much better than random legal play. In a 200-game greedy evaluation, the trained Q-table consistently reached two pegs, while random legal moves averaged more than seven pegs remaining.

Saved training runs have also discovered complete 31-move solutions ending with one peg. These solution files can be replayed with `python replay.py`.

## How It Works

The agent stores Q-values in a dictionary:

```text
q_table[state][action] = expected future value
```

Each state is the flattened 7 by 7 board tuple. Each action is a move tuple:

```text
(from_row, from_col, to_row, to_col)
```

The update rule is standard Q-learning:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
```

The reward function encourages legal progress, lower peg counts, useful board clearing, and solved one-peg endings. Reverse-curriculum training generates easier solvable positions by starting from solved boards and applying random reverse moves.

## Limitations

- The state space is large, so tabular Q-learning requires many episodes.
- The Q-table memorises states and does not generalise like a neural network.
- The exact endgame helper means the strongest training setup is not pure Q-learning.
- Greedy play from the saved Q-table may reach strong near-solutions without always solving the full board.

## Future Improvements

- Add automated tests for moves, rewards, reverse moves, and solved-state checks
- Compare runs with and without curriculum training
- Compare runs with and without the endgame solver
- Add a neural-network value function for better generalisation
- Export experiment summaries as CSV

