# rl-tictactoe 開発メモ（次回の続きはここから）

## 目的
sofcon（社内ソフトコンテスト、社内ゲームAI）とは別の、**強化学習の練習用プロジェクト**。
sofconでMCTS+NN（AlphaZero的パターン）に触れたのをきっかけに、その型を
ゼロから自分の手で組んで理解を深めるのが狙い。DQNは経験済み。
最終的な目標は「NNの熟練者になる」こと（オートエンコーダーなども視野）。

題材は三目並べ（Tic-Tac-Toe, 3x3）。「一番簡単に成果が出るもの」という要望で選定。
状態空間が小さいのでCPU/GPUどちらでも短時間（数分〜）で収束が体感できる。
sofconのコードは一切流用せず、学習のためゼロから実装している。

## 現在の到達点（2026-07-25 時点）
- AlphaZero流パイプライン（自己対戦 → 学習 → vs random評価のループ）を実装済み。
- 動作確認済み：5イテレーション（各20自己対戦、GPU）でlossが低下し、
  vs randomで安定して高勝率（W23〜27 / 30、L0〜2）を確認。バグなしと判断。
- 本番学習（`train/self_play.py` の `main()`、30イテレーション）は**まだ実行していない**。
  checkpointは未保存（`checkpoint/` は空）。

## 構成
```
rl-tictactoe/
├── RESUME.md          このファイル
├── requirements.txt    torch, numpy
├── game/board.py       3x3盤のルール・終局判定
│                        - winner(board): 生の盤面で勝者判定
│                        - terminal_value(board, player): player視点の終局値
│                          （未終局=0, player勝ち=1, 相手勝ち=-1, 引分=1e-4）
│                        - canonical_form(board, player) = board * player
├── model/net.py         小さなCNN。Policy(9)/Value(1)の2ヘッド
├── mcts/mcts.py         PUCT探索。内部は常にcanonical board（手番側=+1）で処理
│                        - alpha-zero-general の実装パターンに準拠（符号反転の作法に注意）
├── train/self_play.py   自己対戦生成 → 学習 → vs random評価、のイテレーションループ
│                        - N_ITERS=30, N_SELFPLAY_GAMES=50, N_SIMS=50 が既定値
│                        - checkpointは checkpoint/net.pt に保存・再開対応
└── play.py              人間 vs AI のテキスト対戦（先手/後手選択可）
```

## すぐ動かすコマンド

### 本番学習を開始
```bash
cd "C:/private/claude/rl-tictactoe" && python train/self_play.py
```
GPU（CUDA）を自動検出して使う。1イテレーションはローカル実測で10〜15秒程度
（GPUはネットが小さいためあまり効かず、MCTSのPython実行がボトルネック）。
`checkpoint/net.pt` があれば自動で読み込んで再開する。

### 人間 vs AI で対戦確認
```bash
cd "C:/private/claude/rl-tictactoe" && python play.py
```

## 次回やること候補（優先順は特になし）
1. 本番学習を実行し、vs randomの勝率・引き分け率の推移を見る
   （三目並べは理論上先手完全読みなら相手のミス以外では負けない → 引き分け/勝ちのみになるはず）
2. MCTSのシミュレーション回数やc_puctを変えて挙動を観察する（AlphaZeroのハイパーパラメータ感覚を掴む）
3. 慣れたら盤を大きくする・別ゲーム（コネクトフォー、6x6オセロなど）に拡張して同じ構造を使い回す
4. 長期的にはオートエンコーダーなど別NNパターンにも手を広げたい（本人の目標）

## ハマりどころメモ
- MCTSの符号反転（negamax式）は間違えやすい。`_search()` は常に
  「このノードの手番から見た価値」を計算し、`return -v` / `return -self.Es[s]`
  で親（＝相手の手番）視点に反転して返す設計（alpha-zero-general準拠）。
  ここを取り違えると学習が全く進まない/逆に弱くなるので、他ゲームに拡張する際も要注意。
- `game/board.py` の `terminal_value(board, player)` はraw board・canonical board
  どちらにも使える汎用実装（符号の意味さえ揃っていればOK）。
