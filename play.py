import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import board as bd
from model.net import PolicyValueNet
from mcts.mcts import MCTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "net.pt")


def main():
    net = PolicyValueNet().to(DEVICE)
    if os.path.exists(CKPT_PATH):
        net.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"loaded checkpoint: {CKPT_PATH}")
    else:
        print("checkpointが見つかりません。未学習のネットで対戦します。")
    net.eval()

    human = None
    while human not in ("1", "2"):
        human = input("先手(X)=1, 後手(O)=2 を選んでください: ").strip()
    human_player = 1 if human == "1" else -1

    board = bd.init_board()
    player = 1
    mcts = MCTS(net, DEVICE, n_sims=200)

    print(bd.render(board))
    while True:
        if player == human_player:
            valid = bd.valid_moves(board)
            move = None
            while move not in valid:
                try:
                    move = int(input(f"手を選んでください (0-8, 空きマス={valid}): ").strip())
                except ValueError:
                    continue
        else:
            pi = mcts.get_action_probs(board, player, temp=0)
            move = int(np.argmax(pi))
            print(f"AI: {move}")

        board = bd.next_state(board, player, move)
        print(bd.render(board))

        r = bd.terminal_value(board, player)
        if r != 0:
            if abs(r - 1e-4) < 1e-6:
                print("引き分け")
            elif player == human_player:
                print("あなたの勝ち！")
            else:
                print("AIの勝ち")
            break
        player = -player


if __name__ == "__main__":
    main()
