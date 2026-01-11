# Gakumasu Character Classifier

CLIPによる画像特徴抽出とKeras 3（PyTorchバックエンド）を組み合わせた画像分類テンプレートです。
少数の学習データで高精度に動作することが期待できます。学園アイドルマスター（学マス）を例としていますが、汎用的な画像分類に利用可能です。

## 環境構築

### 仮想環境の作成と有効化

```bash
python -m venv .my_venv
source .my_venv/bin/activate
```

### 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### データの準備

`data/raw/` 配下に分類したい名前のフォルダ（例: china, hiro, ume）を作成し、画像（png or jpeg形式）を格納します。

### 前処理（特徴量抽出）

画像を512次元のベクトルに変換し、保存します。

```bash
python preprocess.py
```

### 学習

Kerasモデルを訓練し、保存します。

```bash
python train.py
```

### 推論

`test_images/` 内の画像に対して判定を行います。

```bash
python predict.py
```

## ディレクトリ構成

```text
.
├── .my_venv/          # Python仮想環境
├── data/
│   ├── raw/           # [入力] クラス名（アイドル名等）のフォルダに画像を格納
│   └── processed/     # [出力] 特徴量データ(npz)と学習済みモデル(keras)
├── test_images/       # [任意] 推論させたい未知画像を格納
├── utils.py           # CLIPエンコーダー（共通モジュール）
├── preprocess.py      # 画像のベクトル化とデータセット作成
├── train.py           # モデルの学習と保存
├── predict.py         # 未知画像に対する推論実行
└── requirements.txt   # 依存ライブラリ一覧
```
