from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


# ==========================================
# ① Feature Extractor (CLIPの責務)
# ==========================================
class ClipEncoder:
    def __init__(self):
        # 実行環境に合わせて最適なアクセラレータを選択
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.to(self.device).eval()

    def encode(self, image_path: Path):
        """1枚の画像を512次元のベクトルに変換します"""
        img = (
            self.preprocess(Image.open(image_path).convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            feat = self.model.encode_image(img)
            # ベクトルを正規化（長さを1にする）して精度を安定させる
            feat /= feat.norm(dim=-1, keepdim=True)

        return feat.squeeze(0).cpu().numpy()


# ==========================================
# ② Dataset Builder (整理・変換の責務)
# ==========================================
def build_gakumasu_dataset(raw_dir: Path, encoder: ClipEncoder):
    """ディレクトリを巡回してデータセットを構築します"""
    # フォルダ名（アイドル名）を自動取得してソート
    # .startswith('.') を除外することで .DS_Store などを回避
    classes = sorted(
        [d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    X, y = [], []

    for label_id, class_name in enumerate(classes):
        class_dir = raw_dir / class_name
        print(f"Processing {class_name}...")

        # 画像ファイルを一括取得（大文字小文字を問わない glob）
        image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(class_dir.glob(ext))

        for path in image_paths:
            vector = encoder.encode(path)
            X.append(vector)
            y.append(label_id)

    return np.array(X), np.array(y), classes


# ==========================================
# ③ Orchestrator (実行・保存の責務)
# ==========================================
if __name__ == "__main__":
    # パスの定義
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_FILE = PROCESSED_DIR / "gakumasu_data.npz"

    # 保存先ディレクトリがなければ作成
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 変換開始
    encoder = ClipEncoder()
    X, y, label_names = build_gakumasu_dataset(RAW_DIR, encoder)

    # 保存
    if len(X) > 0:
        np.savez(OUTPUT_FILE, x=X, y=y, names=label_names)
        print("\n前処理が完了しましたわ！")
        print(f"保存先: {OUTPUT_FILE}")
        print(f"データの形: {X.shape}")  # 例: (162, 512)
        print(f"ラベル: {label_names}")  # ['china', 'hiro', 'ume']
    else:
        print("\n画像が見つかりませんでした。data/raw の中身を確認してくださいまし。")
