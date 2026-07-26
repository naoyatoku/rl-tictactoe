import os
import sys
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import board as bd
from model.net import PolicyValueNet
from mcts.mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoint")
CKPT_PATH = os.path.join(CKPT_DIR, "net.pt")

N_SIMS = 50
N_SELFPLAY_GAMES = 50
N_ITERS = 30
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
TEMP_THRESHOLD = 6  # これ以降は貪欲(temp=0)に切り替え
EVAL_GAMES = 40


def execute_self_play(net):
    examples = []
    board = bd.init_board()
    player = 1
    mcts = MCTS(net, DEVICE, n_sims=N_SIMS)
    history = []
    move_count = 0

    while True:
        temp = 1.0 if move_count < TEMP_THRESHOLD else 0.0
        pi = mcts.get_action_probs(board, player, temp=temp)
        canon = bd.canonical_form(board, player)
        history.append((canon, pi, player))

        action = int(np.random.choice(len(pi), p=pi))
        board = bd.next_state(board, player, action)

        r = bd.terminal_value(board, player)
        if r != 0:
            for canon_b, p, pl in history:
                z = r if pl == player else -r
                examples.append((canon_b, p, z))
            return examples

        player = -player
        move_count += 1


def train_net(net, examples, optimizer):
    net.train()
    boards = torch.tensor(np.array([e[0] for e in examples]), dtype=torch.float32, device=DEVICE).view(-1, 1, 3, 3)
    pis = torch.tensor(np.array([e[1] for e in examples]), dtype=torch.float32, device=DEVICE)
    zs = torch.tensor(np.array([e[2] for e in examples]), dtype=torch.float32, device=DEVICE).view(-1, 1)

    n = boards.size(0)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b, p, z = boards[idx], pis[idx], zs[idx]

            log_p, v = net(b)
            loss_p = -(p * log_p).sum(dim=1).mean()
            loss_v = F.mse_loss(v, z)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b.size(0)
        print(f"    epoch {epoch + 1}/{EPOCHS} loss={total_loss / n:.4f}")


def random_move(board):
    return random.choice(bd.valid_moves(board))


def play_vs_random(net, n_games=EVAL_GAMES, n_sims=N_SIMS):
    wins, draws, losses = 0, 0, 0
    for g in range(n_games):
        net_player = 1 if g % 2 == 0 else -1
        board = bd.init_board()
        player = 1
        mcts = MCTS(net, DEVICE, n_sims=n_sims)
        while True:
            if player == net_player:
                pi = mcts.get_action_probs(board, player, temp=0)
                action = int(np.argmax(pi))
            else:
                action = random_move(board)
            board = bd.next_state(board, player, action)

            r = bd.terminal_value(board, player)
            if r != 0:
                if abs(r - 1e-4) < 1e-6:
                    draws += 1
                elif player == net_player:
                    wins += 1
                else:
                    losses += 1
                break
            player = -player
    return wins, draws, losses


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    net = PolicyValueNet().to(DEVICE)
    if os.path.exists(CKPT_PATH):
        net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"loaded checkpoint from {CKPT_PATH}")

    optimizer = optim.Adam(net.parameters(), lr=LR)

    for it in range(1, N_ITERS + 1):
        t0 = time.time()
        examples = []
        for _ in range(N_SELFPLAY_GAMES):
            examples.extend(execute_self_play(net))

        train_net(net, examples, optimizer)
        torch.save(net.state_dict(), CKPT_PATH)

        w, d, l = play_vs_random(net)
        dt = time.time() - t0
        print(f"iter {it:03d} | examples={len(examples):4d} | vs random: W{w} D{d} L{l} | {dt:.1f}s")


if __name__ == "__main__":
    main()
