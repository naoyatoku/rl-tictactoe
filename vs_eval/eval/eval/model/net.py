import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 3
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


class PolicyValueNet(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        #共通の畳み込み層
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)               #3×3×1 → 3×3×32(ch)             kernel=3*3*[1*32]   tensor
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)        #3×3×32(ch) → 3×3×32(ch)        kernel=3*3*[32*32]
        self.bn1 = nn.BatchNorm2d(channels)                                         # Batch Normalization正規化            
        self.bn2 = nn.BatchNorm2d(channels)                                         # Batch Normalizaiton正規化            

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)                    # ポリシーヘッド :  (3*3*32) -> (3*3)*2     kernel [1*1][32*2]
        self.policy_bn = nn.BatchNorm2d(2)                                          #                   Batch 正規化
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)        #                   Linear線形結合: 2*3*3(18) -> 9  

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)                     #   評価値ヘッド (32*[3*3]) -> (1*[3*3])  kernel [1*1][32*1]
        self.value_bn = nn.BatchNorm2d(1)                                           #   正規化                           [1*[3*3]]
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 32)                     #   Linear 線形結合                  [3*3] -> 32    (一度たくさんに分散してrelu：非線形関数を挟む)
                                                                                    #   ことで、)評価値をより複雑に表現できるようになる。らしい。   32 -> 1
                                                                                    #   この段数を増やすと、
        self.value_fc2 = nn.Linear(32, 1)                                           #   Linear 線形結合                  batch*32 -> batch * 1

    #順伝播です。
    #
    def forward(self, x):
        # x: (batch, 1, 3, 3)
        x = F.relu(self.bn1(self.conv1(x)))     #conv1 : x(batch * 1*3*3) -> kernel(3×3) -> batch ×32 ×[ 3×3] に分解   → BN → Relu    (batch×32× 3×3 )
        x = F.relu(self.bn2(self.conv2(x)))     #conv2 :          -> kernel(3×3) -> 3×3×32(ch)を維持する。              → BN → Relu    (batch×32× 3×3 )
        #ここで [batch * 32 * 3 * 3 ]の特徴マップができる。]

        #ポリシーヘッド
        p = F.relu(self.policy_bn(self.policy_conv(x)))             #   policy_conv:             x(batch*32* 3*3)->conv->batch * 2* 3*3 ->正規化 -> Relu
        p = p.view(p.size(0), -1)                                   #   reshape（形を変える）    pの0次元目(batch)のサイズは維持し、1次元目以降は全部1つのベクトル(2*3*3=>18次元ベクトル)
        p = self.policy_fc(p)                                       #   policy_fc（線形結合→9)
        log_p = F.log_softmax(p, dim=1)                             #   log_p : log_softmax( 9 )     softmaxはexpを取ると、値が小さくなりすぎたりするのでlogをとるらしい。

        #評価値ヘッド
        v = F.relu(self.value_bn(self.value_conv(x)))               #   x : (batch*3*3)→val_conv(batch*32*3*3)->(batch * 1 * 3*3) -> 正規化 -> relu ( batch * 1 * 3*3)
        v = v.view(v.size(0), -1)                                   #   reshape:2次元化(batch,そのほかは平坦化する)：   batch * 9
        v = F.relu(self.value_fc1(v))                               #   (batch * 9)---->線形結合---> (batch * 32) 一度32個に増やし---> relu
        v = torch.tanh(self.value_fc2(v))                           #   32 --> 線形結合 --> 1 ---> tanh 

        return log_p, v
