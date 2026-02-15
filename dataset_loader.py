from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from config import Config

# ===========================
# 🔧 DATAKÄLLA
# ===========================
# Sökväg till mapp som innehåller frames_animation1–10
DATA_DIR: Path = Path(__file__).parent / Config.DATA_DIR

# ===========================
# 📐 BILDINSTÄLLNINGAR
# ===========================
IMG_SIZE: int = Config.IMG_SIZE
NUM_FRAMES: int = Config.NUM_FRAMES
NUM_ANIMATIONS: int = Config.NUM_ANIMATIONS

# ===========================
# 🔄 BILDTRANSFORMERING
# ===========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


# ===========================
# 🧩 LADDA EN ANIMATIONSMAPP
# ===========================
def load_animation_folder(folder_path: Path | str) -> torch.Tensor:
    """Ladda alla PNG-bilder från en animationsmapp som en tensor.

    Läser bilder i sorterad ordning, applicerar transform
    (resize, grayscale, tensor) och stackar dem.

    Args:
        folder_path: Sökväg till mapp med PNG-frames.

    Returns:
        Tensor med shape [N, C, H, W] där N är antal frames.
    """
    folder_path = Path(folder_path)
    images: list[torch.Tensor] = []
    filenames = sorted(folder_path.iterdir())

    for filepath in filenames:
        if filepath.suffix == '.png':
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

    return torch.stack(images)


# ===========================
# 📦 LADDA HELA DATASETET
# ===========================
def load_dataset() -> torch.Tensor:
    """Ladda hela animationsdatasetet från disk.

    Itererar genom alla animationsmappar (frames_animation1..N),
    laddar varje mapp och stackar till en enda tensor.

    Returns:
        Tensor med shape [NUM_ANIMATIONS, NUM_FRAMES, C, H, W].
    """
    dataset: list[torch.Tensor] = []

    for i in range(1, NUM_ANIMATIONS + 1):
        folder_name = f'frames_animation{i}'
        folder_path = DATA_DIR / folder_name
        print(f'Laddar: {folder_name}')
        animation_tensor = load_animation_folder(folder_path)
        dataset.append(animation_tensor)

    full_tensor = torch.stack(dataset)  # [10, 180, 1, 64, 64]
    print("✅ Dataset färdigladdat. Form:", full_tensor.shape)
    return full_tensor


# ===========================
# 🚀 KÖR SOM SCRIPT
# ===========================
if __name__ == '__main__':
    dataset = load_dataset()
