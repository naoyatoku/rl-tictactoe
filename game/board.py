import numpy as np

N = 3
ACTION_SIZE = N * N


def init_board():
    return np.zeros((N, N), dtype=np.int8)


def valid_moves(board):
    flat = board.flatten()
    return [i for i in range(ACTION_SIZE) if flat[i] == 0]


def valid_moves_mask(board):
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for i in valid_moves(board):
        mask[i] = 1.0
    return mask


def next_state(board, player, action):
    b = board.copy()
    b.flat[action] = player
    return b


def _lines(board):
    rows = [board[i, :] for i in range(N)]
    cols = [board[:, i] for i in range(N)]
    diags = [np.diag(board), np.diag(np.fliplr(board))]
    return rows + cols + diags


def winner(board):
    for line in _lines(board):
        s = int(line.sum())
        if s == N:
            return 1
        if s == -N:
            return -1
    return 0


def terminal_value(board, player):
    """board上の状態を`player`の視点で評価する。
    未終局なら0、player側のラインが揃っていれば1、
    相手側が揃っていれば-1、引き分けなら小さな非ゼロ値を返す。
    """
    w = winner(board)
    if w != 0:
        return 1 if w == player else -1
    if len(valid_moves(board)) == 0:
        return 1e-4
    return 0


def canonical_form(board, player):
    return board * player


def render(board):
    symbols = {1: "X", -1: "O", 0: "."}
    lines = []
    for r in range(N):
        lines.append(" ".join(symbols[int(v)] for v in board[r]))
    return "\n".join(lines)


def board_key(board):
    return board.tobytes()
