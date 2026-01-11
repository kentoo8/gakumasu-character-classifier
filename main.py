import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
model = model.to(device).eval()  # evalが大事です :contentReference[oaicite:2]{index=2}


def image_embedding(path: str) -> torch.Tensor:
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)  # (1, D) :contentReference[oaicite:3]{index=3}
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()  # (D,)
