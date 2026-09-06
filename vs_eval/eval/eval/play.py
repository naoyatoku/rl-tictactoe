import os
import sys

import numpy as np
import torch
#
import random


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
#        human = input("先手(X)=1, 後手(O)=2 を選んでください: ").strip()
        human="1"
    human_player = 1 if human == "1" else -1

    board = bd.init_board()
    player = 1
    mcts = MCTS(net, DEVICE, n_sims=200)
    print(bd.render(board))
    while True:
        if player == human_player:
            print(f"---------- [P] ----------[{player}]")
            valid = bd.valid_moves(board)
            move = random.choice(valid)
        else:
            print(f"---------- [AI] ---------[{player}]")
            pi = mcts.get_action_probs(board, player, temp=0)       #これがMTCSで手を見つける
            move = int(np.argmax(pi))
            print(f"AI: {move}")

        board = bd.next_state(board, player, move)
        print(bd.render(board))

        #勝負がついている場合は、自分勝ち = 1 , 自分負け = -1 , もう置く場所がない = e^-4(ひきわけ)  , まだ勝負ついてない = 0
        r = bd.winner(board)
#        r = bd.terminal_value(board)
        if r != 0:                                  #なんらかしらの勝負がついた
            if r== human_player:
                print("you win!")
            else:
                print("AI win!")
            break
        else:                                       #勝負はついてないが、盤面がいっぱいで置く場所がない場合
            if len(bd.valid_moves(board)) == 0:
                print("draw!")
                break
        player = -player                            #   1 と -1を入れ替わる
if __name__ == "__main__":
    main()
