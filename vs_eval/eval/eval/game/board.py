import numpy as np

N = 3
ACTION_SIZE = N * N


def init_board():
    return np.zeros((N, N), dtype=np.int8)


#おいてもいい場所を列挙する。
def valid_moves(board):
    flat = board.flatten()          #3×3の配列を1字配列に変換
    #if flat[i]==0の条件にあっている値だけ、
    a =  [i for i in range(ACTION_SIZE) if flat[i] == 0]    #debug code : 
    return [i for i in range(ACTION_SIZE) if flat[i] == 0]


def valid_moves_mask(board):
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for i in valid_moves(board):
        mask[i] = 1.0                   #valid_movesで0(おける場所)のインデックスの場所を1でマスクする。
    return mask


def next_state(board, player, action):
    b = board.copy()
    b.flat[action] = player
    return b


def _lines(board):
    rows = [board[i, :] for i in range(N)]                  #横の「行」のパターンを全部取り出す
    cols = [board[:, i] for i in range(N)]                  #縦の「列」のパターンを全部取り出す
    diags = [np.diag(board), np.diag(np.fliplr(board))]     #対角左上→右下、右上→左下へのパターンを取り出す
    return rows + cols + diags                              #全部のパターンをつなげて返す。


def winner(board):
    for line in _lines(board):                              #_linesは、縦、横、斜めの場所の数値配列を全部取り出す
        s = int(line.sum())                                 #全部の要素を足す。（1が全部ならN、-1が全部なら-Nになる)
        if s == N:
            return 1                                        #プレイヤー1が全部そろっている
        if s == -N:                                         #プレイヤー-1が全部そろっている。
            return -1
    return 0                                                #そろってない

def game_result(board):
    """盤面だけを見た客観的な結果。
    None = 未終局 / 1 = プレイヤー1の勝ち / -1 = プレイヤー-1の勝ち / 0 = 引き分け
    """
    w = winner(board)
    if w != 0:
        return w
    if not valid_moves(board):
        return 0
    return None
    #これはplayer視点での価値を返す関数
    
def terminal_value(board, player):
    """board上の状態を`player`の視点で評価する。
    未終局なら0、player側のラインが揃っていれば1、
    相手側が揃っていれば-1、引き分けなら小さな非ゼロ値を返す。
    """
    w = winner(board)
    if w != 0:                              #   勝っている人がいる場合、それがplayerなら1を返す
        return 1 if w == player else -1     #   
    if len(valid_moves(board)) == 0:        #   動ける場所がない場合、
        return 1e-4                         #   ほぼ0を返すこれはこれでゲームオーバー？
    return 0                                #   勝負がついてない場合、0を返す


#playerの数値を全配列の数値に
def canonical_form(board, player):
    return board * player


def render(board,indent=0):
    symbols = {1: "X", -1: "O", 0: "."}
    prefix = "  " * indent          # ← 追加。深さ×4スペース分の余白を作る
    lines = []
    for r in range(N):
        lines.append(prefix + "".join(symbols[int(v)] for v in board[r]))
    return "\n".join(lines)


def board_key(board):
    return board.tobytes()
