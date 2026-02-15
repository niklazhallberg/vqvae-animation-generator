# --- START OF FILE start_training.py (Anpassad för Config-integration) ---

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.vqvae_model import VQVAE
from train_vqvae import train_vqvae
from dataset_loader import load_dataset
from config import Config  # <--- HÄR IMPORTERAS DEN NYA KONFIGURATIONEN

# =======================
# ✨ GLOBAL SEEDING
# =======================
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

# =======================
# ⚙️ Sökvägar (Hämtas nu från Config)
# =======================
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Config.IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# 📦 Ladda dataset
# =======================
print("Laddar dataset...")
data_tensor_anims = load_dataset()
data_tensor_flat = data_tensor_anims.view(-1, 1, Config.IMG_SIZE, Config.IMG_SIZE)
print(f"Dataset omformat: {data_tensor_flat.shape}")
full_dataset = TensorDataset(data_tensor_flat)

# =======================
# 🧪 Skapa Train/Val Loaders
# =======================
train_size = int(Config.TRAIN_VAL_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(Config.SEED)
)

train_loader = DataLoader(
    train_data, batch_size=Config.BATCH_SIZE, shuffle=True,
    num_workers=Config.NUM_WORKERS, generator=torch.Generator().manual_seed(Config.SEED)
)
val_loader = DataLoader(
    val_data, batch_size=Config.BATCH_SIZE, shuffle=False,
    num_workers=Config.NUM_WORKERS
)

# =======================
# 🧠 Initiera modell (Parametrar från Config)
# =======================
device = Config.DEVICE
print(f"Använder device: {device}")

model = VQVAE(
    embedding_dim=Config.EMBEDDING_DIM,
    num_embeddings=Config.NUM_EMBEDDINGS,
    commitment_cost=Config.BETA_START,
    ema_decay=Config.EMA_DECAY,
    ema_epsilon=Config.EMA_EPSILON,
    ema_recovery_threshold=Config.RECOVERY_THRESHOLD,
    ema_recovery_probability=Config.RECOVERY_PROB,
    ema_recovery_noise_scale=Config.RECOVERY_NOISE_SCALE
).to(device)

print(f"Modell initierad: {sum(p.numel() for p in model.parameters())} parametrar.")

# =======================
# 🏋️ Starta träning (Allt styrs nu via Config-objektet)
# =======================
print(f"\nStarting Training with Spatial VQ-VAE (Config-Driven)")
print(f"Hardware: {device}")
print(f"Recovery: Threshold={Config.RECOVERY_THRESHOLD}, Prob={Config.RECOVERY_PROB}")

train_vqvae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=Config.EPOCHS,
    learning_rate=Config.LR,
    beta_start=Config.BETA_START,
    beta_end=Config.BETA_END,
    beta_epochs=Config.BETA_EPOCHS,
    patience=Config.PATIENCE,
    min_delta=Config.MIN_DELTA,
    codebook_loss_weight=Config.CODEBOOK_LOSS_WEIGHT,
    checkpoint_dir=Config.CHECKPOINT_DIR,
    log_dir=Config.LOG_DIR,
    output_dir=Config.IMAGE_OUTPUT_DIR,
    vis_epoch_interval=Config.VIS_EPOCH_INTERVAL,
    log_interval=Config.LOG_INTERVAL,
    force_kmeans_init=Config.FORCE_KMEANS_INIT,
    scheduler_factor=Config.SCHEDULER_FACTOR,
    scheduler_min_lr=Config.SCHEDULER_MIN_LR,
    num_vis_examples=Config.NUM_VIS_EXAMPLES,
)

print("\nTräning avslutad.")
