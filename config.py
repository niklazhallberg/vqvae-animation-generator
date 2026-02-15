# --- START OF FILE config.py ---
from dataclasses import dataclass, field
from pathlib import Path

import torch


# =====================================================
# 📁 FILSYSTEM & DATA
# =====================================================
@dataclass(frozen=True)
class PathConfig:
    """Sökvägar för data, checkpoints, loggar och bilder."""

    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_dir: Path = Path("outputs/logs")
    image_output_dir: Path = Path("outputs/images")


# =====================================================
# 📐 BILDKONFIGURATION
# =====================================================
@dataclass(frozen=True)
class DataConfig:
    """Inställningar för bilddata och dataset."""

    img_size: int = 64
    in_channels: int = 1          # Grayscale från p5.js
    num_frames: int = 180
    num_animations: int = 10
    train_val_split: float = 0.9


# =====================================================
# 📐 ARKITEKTUR (VQ-VAE)
# =====================================================
@dataclass(frozen=True)
class ModelConfig:
    """Arkitekturparametrar för VQ-VAE-modellen."""

    embedding_dim: int = 64       # Storlek på varje vektor i kodboken
    num_embeddings: int = 128     # Antal "visuella ord" i kodboken


# =====================================================
# 🏋️ TRÄNINGSINSTÄLLNINGAR
# =====================================================
@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparametrar för träning, stabilitet och anti-collapse."""

    batch_size: int = 32
    epochs: int = 200
    lr: float = 2e-4
    seed: int = 42

    # Beta-Warmup (Commitment Loss)
    beta_start: float = 0.05
    beta_end: float = 0.25
    beta_epochs: int = 30         # Hur många epoker rampen pågår

    # EMA (Exponential Moving Average)
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    # Dead Code Recovery
    recovery_threshold: float = 1e-6  # Tröskel för "döda" koder
    recovery_prob: float = 0.1        # Sannolikhet för återhämtning per pass
    recovery_noise_scale: float = 0.02

    # Initialization
    force_kmeans_init: bool = True    # Starta med "lexikon" från datan

    # Early stopping & monitoring
    patience: int = 15
    min_delta: float = 1e-4
    log_interval: int = 50
    vis_epoch_interval: int = 5       # Hur ofta vi sparar exempelbilder

    # LR Scheduler
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # DataLoader
    num_workers: int = 0

    # Codebook loss weight (0.0 för EMA)
    codebook_loss_weight: float = 0.0

    # Visualisering
    num_vis_examples: int = 8


# =====================================================
# ⚙️ SYSTEM & DEVICE
# =====================================================
@dataclass(frozen=True)
class DeviceConfig:
    """Enhetskonfiguration med MPS -> CUDA -> CPU fallback."""

    device: torch.device = field(default_factory=lambda: torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    ))


# =====================================================
# 🔗 SAMMANSLAGEN KONFIGURATION (bakåtkompatibel)
# =====================================================
class Config:
    """Bakåtkompatibelt konfigurationsobjekt som samlar alla delkonfigurationer."""

    _paths = PathConfig()
    _data = DataConfig()
    _model = ModelConfig()
    _training = TrainingConfig()
    _device = DeviceConfig()

    # 📁 Sökvägar
    DATA_DIR: Path = _paths.data_dir
    CHECKPOINT_DIR: Path = _paths.checkpoint_dir
    LOG_DIR: Path = _paths.log_dir
    IMAGE_OUTPUT_DIR: Path = _paths.image_output_dir

    # 📐 Bilddata
    IMG_SIZE: int = _data.img_size
    IN_CHANNELS: int = _data.in_channels
    NUM_FRAMES: int = _data.num_frames
    NUM_ANIMATIONS: int = _data.num_animations
    TRAIN_VAL_SPLIT: float = _data.train_val_split

    # 📐 Arkitektur
    EMBEDDING_DIM: int = _model.embedding_dim
    NUM_EMBEDDINGS: int = _model.num_embeddings

    # 🏋️ Träning
    BATCH_SIZE: int = _training.batch_size
    EPOCHS: int = _training.epochs
    LR: float = _training.lr
    SEED: int = _training.seed
    BETA_START: float = _training.beta_start
    BETA_END: float = _training.beta_end
    BETA_EPOCHS: int = _training.beta_epochs
    EMA_DECAY: float = _training.ema_decay
    EMA_EPSILON: float = _training.ema_epsilon
    RECOVERY_THRESHOLD: float = _training.recovery_threshold
    RECOVERY_PROB: float = _training.recovery_prob
    RECOVERY_NOISE_SCALE: float = _training.recovery_noise_scale
    FORCE_KMEANS_INIT: bool = _training.force_kmeans_init
    PATIENCE: int = _training.patience
    MIN_DELTA: float = _training.min_delta
    LOG_INTERVAL: int = _training.log_interval
    VIS_EPOCH_INTERVAL: int = _training.vis_epoch_interval
    SCHEDULER_FACTOR: float = _training.scheduler_factor
    SCHEDULER_MIN_LR: float = _training.scheduler_min_lr
    NUM_WORKERS: int = _training.num_workers
    CODEBOOK_LOSS_WEIGHT: float = _training.codebook_loss_weight
    NUM_VIS_EXAMPLES: int = _training.num_vis_examples

    # ⚙️ Device
    DEVICE: torch.device = _device.device
