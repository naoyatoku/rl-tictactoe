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
        #toku 追加した
        self.dep =0      #

    #================================================================================
    #   predictは、1(バッチサイズ)×1(チャンネル)×3×3(盤面)
    #   
    #   
    #   
    #================================================================================
    def _predict(self, canon_board):
        x = torch.tensor(canon_board, dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.net.eval()                         #   nn.Module 推論モードらしい。
        with torch.no_grad():                   #   計算グラフの記録をしない呪文らしい。
            log_p, v = self.net(x)                  #   ※no_grad()は、オブジェクトで__enter__と__exit__を備えている必要があり、それをブロックの先頭と出口で呼び出す。
                                                #   net()は、net class(class PolicyValueNe) の __call__メソッドの呼び出し。
                                                #   nn.Module の __call__ ではforwardが呼ばれます。
                                                #   ここでx = 盤面について、推論したNetの結果[1,9](9マス分の確率分布:softmax)の値を返す.
                                                #   vが盤面の評価値みたいなやつ。

        p = torch.exp(log_p).cpu().numpy().flatten()    #log_p[1,9] → log_softmax  なので、expをとって戻すようなイメージ。pが確率分布。
        # p = 
        return p, v.item()      #v(評価値)とp（確率分布）

    #================================================================================
    #   playerが自分の手番で可変だが、canonical formは常に自分が「1」になるように変換される。
    #   これ以降は自分はplayer=1として扱う。
    #================================================================================
    def get_action_probs(self, board, player, temp=1.0):
        #   canonical_formというのが、boardを、常に自分が「1」敵が「-1」だとして学習させるために、
        #   playerをかけてplayerを[1]目線で
        canon = bd.canonical_form(board, player)            #自分を1とした盤面に変換する。
        for _ in range(self.n_sims):        #   n_simsが試行回数。
            print(f"mtcs : sim[{_}]")
            self.dep = -1     #深さを-1としておきます。
            self._search(canon)             #   self._search()の戻り値は使われてない.
                                            #   _search()の中で、Qsa, Nsa, Ps, Ns, valid, Esが更新される。

        s = bd.board_key(canon)             #sが現状盤面の
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
        #入った時に、深さを一つ足します。
        self.dep+=1
        s = bd.board_key(canon_board)           #イミュータブルな盤面状態を作る。

        #===============================================================================
        #   toku 盤面の表示をします。
        #==============================================================================
        print(bd.render(canon_board,self.dep))

        #===============================================================================
        #   Esにsが初めて出てきた場合は、Es[s]に勝負がついているかどうかを入れる。
        #   ※Es[s] は、終了ステータス(自分が勝ち=1 ,相手が勝ち=-1, 引き分け=e^-4, 勝負がついてない=0 )。
        #===============================================================================
        if s not in self.Es:
            #   これむずいけど、辞書型の配列Esのキー「s：現在の盤面のイミュータブル版」→sは普遍なのでキーになれる
            #   Es[s]には、この結果 canon_board での結果(自分が勝ち=1 ,相手が勝ち=-1, 引き分け=e^-4, 勝負がついてない=0 )
            #   を入れる。
            #=========================================================================================+		special variables		

            #   terminal valueには player=1 とした結果を返す。
            #=========================================================================================
            self.Es[s] = bd.terminal_value(canon_board, 1)
        #===============================================================================
        #   Es[s]が勝負がついている場合、
        #   この盤面を作った人（手前の手番の人）が勝ったかどうかの価値を返す。
        #===============================================================================
        if self.Es[s] != 0:
            self.dep-=1
            return -self.Es[s]      #この盤面を作ったのは手前の手番の人なので、その人にとっても価値を返す。

        #===============================================================================
        #  Ps[s]が初めて出てきた場合、NNのPredict(forward)を行う
        #===============================================================================
        if s not in self.Ps:
            p, v = self._predict(canon_board)
            mask = bd.valid_moves_mask(canon_board)                 #   maskは、[1,9]配列で、おける場所を1、おけない場所を0とした配列
            p = p * mask                                            #   p(方策)について、おける場所だけにする。配列*配列は、それぞれの要素を掛け算するようだ
            total = p.sum()                                         #   おける場所の価値をすべて足したもの
            p = p / total if total > 0 else mask / mask.sum()       #   totalで正規化です v  = v / total
            self.Ps[s] = p                                          #   Ps[s]にこれを入れる(全9マス分の)
            self.valid[s] = mask                                    #   おいていい場所いけない場所を
            self.Ns[s] = 0                                          #   Nsは、訪問回数
            self.dep-=1
            return -v
        #================================================================================
        #   Nsa[s,a]    :   回数,
        #   Qsa[s,a]    :   価値
        #   PsやEsがすでにある場合にここに来る。
        #================================================================================
        mask = self.valid[s]
        best_u, best_a = -float("inf"), -1                      #
        for a in range(bd.ACTION_SIZE):                         #   取りうるアクションで、一番価値の高いものを探す
            if mask[a] == 0:
                continue
            q = self.Qsa.get((s, a), 0.0)                       #   self. Qsaは、状態[s]、[a]という手を使ったときの価値を格納する辞書。なければ0,0を使う。
            nsa = self.Nsa.get((s, a), 0)                       #   self.Nsaは、状態[s]で、[a]という手を使った回数を数える辞書。見つからない場合は0を使う
            #------------------------------------------------------------------------------------------------------
            #   価値計算    :   PUCT計算
            #   q                 :   現在までの価値
            #   puct              :   探索をどれだけ重視するのか(1.5固定)
            #   Ps[s][a]          :   Psは、predictのpolicy；NNの選択した結果(各マス目の有効確率分布:0-8配列)
            #   sqrt(Ns[s])       :   √探索回数   ,
            #       1+nsa         :  この手を試した回数が増えると大きくなる(割り算すると)回数
            #   -------->   q 以降は探索ボーナス。同じ盤面が何度も起こると、探索ボーナス自体は低くなるが、
            #               たくさん試行していくうちに、実際の価値の値が浮き出てくるようなイメージ
            #   Ps[s][a]        :   実際、学習していくうち、この値が的確になっていくのかな？
            #------------------------------------------------------------------------------------------------------
            u = q + self.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8) / (1 + nsa)  #
            if u > best_u:
                best_u, best_a = u, a

        a = best_a          #今回選択するべき手。
        next_canon = bd.next_state(canon_board, 1, a)       #   aを打ったときの盤面
        next_canon = bd.canonical_form(next_canon, -1)      #   次の人から見た盤面(1がプレイヤーとなる)

        v = self._search(next_canon)                        #   次の人（相手が打ったベストな手の)の価値の「反転」
        
        if (s, a) in self.Qsa:                              #   Qsa（価値アレイ）に[s.a]があるときは
            nsa = self.Nsa[(s, a)]                                      
            self.Qsa[(s, a)] = (nsa * self.Qsa[(s, a)] + v) / (nsa + 1)     #Qsaに探索回数に応じた値にして足す
            self.Nsa[(s, a)] += 1                                           #[s,a]の探索回数
        else:
            self.Qsa[(s, a)] = v                                            #Qsaを新設します。
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        self.dep-=1
        return -v       #結局このvは、NNのv若しくは終局の価値
