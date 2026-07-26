import math
import numpy as np
import torch

from game import board as bd

C_PUCT = 1.5


class MCTS:
    """AlphaZero流のPUCT探索。全ての内部状態はcanonical board（手番側が常に+1）で扱う。"""

    def __init__(self, net, device, n_sims=100, c_puct=C_PUCT):
        self.net = net
        self.device = device
        self.n_sims = n_sims
        self.c_puct = c_puct

        self.Ps = {}    # s -> 方策の事前分布 (numpy array)
        self.Ns = {}    # s -> 訪問回数の合計
        self.Nsa = {}   # (s, a) -> 訪問回数
        self.Qsa = {}   # (s, a) -> 平均価値
        self.valid = {}  # s -> 合法手マスク
        self.Es = {}    # s -> 終局判定結果（手番視点、0なら未終局）

    def _predict(self, canon_board):
        x = torch.tensor(canon_board, dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.net.eval()
        with torch.no_grad():
            log_p, v = self.net(x)
        p = torch.exp(log_p).cpu().numpy().flatten()
        return p, v.item()

    def get_action_probs(self, board, player, temp=1.0):
        canon = bd.canonical_form(board, player)
        for _ in range(self.n_sims):
            self._search(canon)

        s = bd.board_key(canon)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(bd.ACTION_SIZE)], dtype=np.float32)

        if temp == 0:
            probs = np.zeros_like(counts)
            best = int(np.argmax(counts))
            probs[best] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total == 0:
            mask = bd.valid_moves_mask(canon)
            return mask / mask.sum()
        return counts / total

    def _search(self, canon_board):
        s = bd.board_key(canon_board)

        if s not in self.Es:
            self.Es[s] = bd.terminal_value(canon_board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            p, v = self._predict(canon_board)
            mask = bd.valid_moves_mask(canon_board)
            p = p * mask
            total = p.sum()
            p = p / total if total > 0 else mask / mask.sum()
            self.Ps[s] = p
            self.valid[s] = mask
            self.Ns[s] = 0
            return -v

        mask = self.valid[s]
        best_u, best_a = -float("inf"), -1
        for a in range(bd.ACTION_SIZE):
            if mask[a] == 0:
                continue
            q = self.Qsa.get((s, a), 0.0)
            nsa = self.Nsa.get((s, a), 0)
            u = q + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + nsa)
            if u > best_u:
                best_u, best_a = u, a

        a = best_a
        next_canon = bd.next_state(canon_board, 1, a)
        next_canon = bd.canonical_form(next_canon, -1)

        v = self._search(next_canon)

        if (s, a) in self.Qsa:
            nsa = self.Nsa[(s, a)]
            self.Qsa[(s, a)] = (nsa * self.Qsa[(s, a)] + v) / (nsa + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
