# utils.py
from pathlib import Path

import open_clip
import torch
from PIL import Image


class ClipEncoder:
    """
    CLIP (Contrastive Language-Image Pre-training) モデルを用いて
    画像から高次元の特徴量ベクトルを抽出するクラス。

    画像内の視覚的な情報を、意味的な特徴を保持したまま512次元の数値配列に変換します。

    能力:
        - 画像の概念的特徴（形状、色、雰囲気など）の数値化
        - 各種アクセラレータ（MPS/CUDA）への自動対応
        - 特徴ベクトルのL2正規化による出力の標準化
    """

    def __init__(self):
        # 実行環境に合わせて最適なアクセラレータを選択
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"ClipEncoder initialized on: {self.device}")

        # モデルと前処理関数のロード
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.to(self.device).eval()

    def encode(self, image_path: Path):
        """画像を読み込み、正規化された512次元ベクトルを返します"""
        img = (
            self.preprocess(Image.open(image_path).convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            feat = self.model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)

        return feat.squeeze(0).cpu().numpy()
