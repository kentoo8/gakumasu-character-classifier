import os
import warnings
from pathlib import Path

import numpy as np
from utils import ClipEncoder

# KerasバックエンドをPyTorchに固定 (MPS/GPUアクセラレーション有効化)
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore", category=FutureWarning)

import keras  # noqa: E402


def load_resources(model_path: Path, data_path: Path):
    """推論に必要なモデルとラベル情報をロードします。"""
    if not model_path.exists() or not data_path.exists():
        raise FileNotFoundError(
            "学習済みモデルまたはデータセット（npz）が見つかりません。"
        )

    # モデルのロード
    model = keras.models.load_model(str(model_path))

    # クラス名（アイドル名）の対応表を npz から復元
    data = np.load(data_path, allow_pickle=True)
    class_names = data["names"]

    return model, class_names


def run_prediction(image_path: Path, model, encoder, class_names):
    """
    1枚の画像に対して推論を行い、結果をコンソールに表示します。
    """
    # 1. CLIPで画像をベクトル化
    vector = encoder.encode(image_path)

    # 2. モデル入力用に次元を合わせる [1, 512]
    input_data = np.expand_dims(vector, axis=0)

    # 3. 推論実行
    # batch_size=1 の推論。verbose=0 で不要なログを抑制
    preds = model.predict(input_data, verbose=0)

    # 4. 最も確率の高いインデックスとその確率を取得
    best_idx = np.argmax(preds[0])
    confidence = preds[0][best_idx] * 100
    label = class_names[best_idx]

    print(
        f"Result: 【 {label: <8} 】 (Confidence: {confidence:6.2f}%) "
        f"| File: {image_path.name}"
    )


if __name__ == "__main__":
    # パス設定
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "data" / "processed" / "idol_model.keras"
    DATA_PATH = BASE_DIR / "data" / "processed" / "gakumasu_data.npz"
    TEST_DIR = BASE_DIR / "test_images"

    # 1. 準備（モデル、エンコーダーの初期化）
    try:
        model, class_names = load_resources(MODEL_PATH, DATA_PATH)
        encoder = ClipEncoder()
    except Exception as e:
        print(f"初期化エラー: {e}")
        exit()

    # 2. テスト画像の取得
    TEST_DIR.mkdir(exist_ok=True)
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    test_files = []
    for ext in image_extensions:
        test_files.extend(TEST_DIR.glob(ext))

    if not test_files:
        print(
            f"\n提示: {TEST_DIR.relative_to(BASE_DIR)} フォルダに"
            "判定したい画像を入れてくださいまし。"
        )
        exit()

    # 3. 順次判定
    print(f"\n{'=' * 60}")
    print(f"  学マスアイドル判定 (Total: {len(test_files)} images)")
    print(f"{'=' * 60}")

    for img_path in sorted(test_files):
        run_prediction(img_path, model, encoder, class_names)

    print(f"{'=' * 60}\n")
