import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# KerasのバックエンドをPyTorchに設定 (インポート前に実行が必要)
os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings("ignore", category=FutureWarning)


import keras  # noqa: E402
from keras import layers  # noqa: E402


def load_data(data_path: Path):
    """
    preprocess.py で生成した npz ファイルから学習データを展開します。
    """
    if not data_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりませんわ: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    x = data["x"]  # [枚数, 512次元] の特徴量
    y = data["y"]  # 各データの正解ID
    names = data["names"]  # IDに対応する名前リスト

    print("--- Data Loaded ---")
    print(f"Total samples: {x.shape[0]}")
    print(f"Class names: {names}")

    return x, y, names


def build_model(input_dim: int, num_classes: int):
    """
    Kerasモデルを構築し、コンパイルして返します。

    Args:
        input_dim (int): 入力データの次元数 (CLIPなら512)
        num_classes (int): 分類するアイドルの人数
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),  # 過学習防止
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # パスの設定
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "gakumasu_data.npz"
    MODEL_PATH = BASE_DIR / "data" / "processed" / "idol_model.keras"

    # 1. データのロード
    x, y, class_names = load_data(DATA_PATH)

    # 2. 学習用とテスト用に分割
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. モデルの構築
    # 特徴量の次元数 (512) とアイドルの人数を渡す
    model = build_model(input_dim=x.shape[1], num_classes=len(class_names))
    model.summary()

    # 4. 学習
    print("\nStarting training...")
    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # 5. 評価
    test_scores = model.evaluate(x=x_test, y=y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    # 6. 保存
    # モデル構造、重み、コンパイル情報がすべて1ファイルに保存されます
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"\nTraining complete and model saved to: {MODEL_PATH}")
